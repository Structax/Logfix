use anyhow::{Context, Result};
use rayon::prelude::*;
use std::fs;
use clap::{Arg, Command, ArgAction};
use serde::{Serialize, Deserialize};
use serde_json;
use serde_yaml;
use regex::Regex;
use similar::{TextDiff, Algorithm};
use colored::*;
use reqwest::Client;
use tokio;
use tiktoken_rs::cl100k_base;
use indicatif::{ProgressBar, ProgressStyle};
use std::{thread, time::Duration};



#[derive(Serialize)]
struct LogOutput {
    file: String,
    errors: Vec<String>,
    warnings: Vec<String>,
    infos: Vec<String>,
    debugs: Vec<String>,
    criticals: Vec<String>,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageContent,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

fn show_diff(original: &str, modified: &str) {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_lines(original, modified);
    
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Delete => println!("{}", format!("-{}", change).red()),
            similar::ChangeTag::Insert => println!("{}", format!("+{}", change).green()),
            similar::ChangeTag::Equal => println!("{}", format!(" {}", change).white()),
        }
    }
}
fn colorize_log(log: &str) -> String {
    if log.contains("ERROR") {
        log.red().to_string()
    } else if log.contains("WARNING") {
        log.yellow().to_string()
    } else if log.contains("INFO") {
        log.blue().to_string()
    } else if log.contains("DEBUG") {
        log.green().to_string()
    } else if log.contains("CRITICAL") {
        log.red().bold().to_string()
    } else {
        log.to_string()
    }
}
pub fn optimize_log_data(log_data: &str, max_tokens: usize) -> Result<String> {
    let bpe = cl100k_base()?;
    let tokens = bpe.encode_with_special_tokens(log_data);
    let token_count = tokens.len();

    if token_count <= max_tokens {
        return Ok(log_data.to_string());
    }

    let truncated_tokens = tokens[..max_tokens].to_vec(); // Vec<u32> に変換
    let optimized_log = bpe.decode(truncated_tokens)?; // ✅ そのまま渡す

    Ok(optimized_log)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimize_log_data() {
        let log = "Error: Something went wrong! Please check the system logs for more details.";
        let optimized = optimize_log_data(log, 10).unwrap();
        
        assert!(optimized.len() < log.len());
    }
}

async fn get_fix_suggestion(log: &str, ai_mode: &str) -> Result<String> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner}  {msg}").unwrap());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message("Analyzing log and generating fix...");
    pb.set_message("Analyzing log with AI...");

    let optimized_log = optimize_log_data(log, 4096)?; // 4096トークン以内に制限

    let api_url = "https://api.openai.com/v1/chat/completions";
    let client = Client::new();
    let prompt = if ai_mode == "full" {
        format!(
            "You are an expert software engineer.
            Your task is to analyze the following log file and generate a FIXED version of the source code.
            
            - The programming language is: Rust (or Python, JavaScript, etc.).
            - If the log contains source code with errors, provide the corrected version.
            - If the log only contains error messages without code, suggest possible fixes.
            - If the error cause is unclear, provide general debugging steps.
            - Return only the corrected code or fix suggestions, do not include explanations.
            - Format all responses in a clear and structured way.
    
            Here is the error log:
            ```log
            {}
            ```",
            optimized_log
        )
    } else {
        format!(
            "The following log contains errors. Please generate fixes.
    
            Here is the error log:
            ```log
            {}
            ```",
            optimized_log
        )
    };
    
      
    println!("✅ DEBUG: Sending request to OpenAI API...");
    let response = client.post(api_url)
        .header("Authorization", &format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{ "role": "system", "content": "You are an expert log analyzer." },
                         { "role": "user", "content": prompt }],
            "temperature": 0.7
        }))
        .send()
        .await?;
     pb.finish_with_message("✅ AI Analysis complete!");  
     println!("✅ DEBUG: OpenAI API response received!");

     let response_json = response.json::<serde_json::Value>().await?;
     println!("✅ DEBUG: OpenAI API response: {:#?}", response_json);
        
        if let Some(choice) = response_json["choices"].get(0) {
            if let Some(text) = choice["message"]["content"].as_str() {
                return Ok(text.to_string());
            }
        }
        println!("❌ DEBUG: OpenAI API response did not contain a valid fix.");
        Err(anyhow::anyhow!("Failed to get AI-generated fix"))
    }

   
fn detect_log_format(contents: &str) -> &str {
    if contents.trim().starts_with('{') {
        "json"
    } else if contents.trim().starts_with("---") || contents.contains(":\n") {
        "yaml"
    } else if contents.contains("=") {
        "toml"
    } else {
        "plain"
    }
}

fn parse_log(contents: &str) -> Vec<String> {
    match detect_log_format(contents) {
        "json" => serde_json::from_str::<Vec<String>>(contents).unwrap_or_else(|_| vec![contents.to_string()]),
        "yaml" => serde_yaml::from_str::<Vec<String>>(contents).unwrap_or_else(|_| vec![contents.to_string()]),
        _ => contents.lines().map(String::from).collect(),
    }
}

fn process_logs_by_level(contents: &str) -> LogOutput {
    let error_regex = Regex::new(r"(?i)error[: ](.*)").unwrap();
    let warning_regex = Regex::new(r"(?i)warning[: ](.*)").unwrap();
    let info_regex = Regex::new(r"(?i)info[: ](.*)").unwrap();
    let debug_regex = Regex::new(r"(?i)debug[: ](.*)").unwrap();
    let critical_regex = Regex::new(r"(?i)critical[: ](.*)").unwrap();
    
    let lines = parse_log(contents);
    let (errors, warnings, infos, debugs, criticals) = lines
        .par_iter()
        .map(|line| {
            let mut error = None;
            let mut warning = None;
            let mut info = None;
            let mut debug = None;
            let mut critical = None;

            if let Some(cap) = error_regex.captures(line) {
                error = Some(cap[1].to_string());
            } else if let Some(cap) = warning_regex.captures(line) {
                warning = Some(cap[1].to_string());
            } else if let Some(cap) = info_regex.captures(line) {
                info = Some(cap[1].to_string());
            } else if let Some(cap) = debug_regex.captures(line) {
                debug = Some(cap[1].to_string());
            } else if let Some(cap) = critical_regex.captures(line) {
                critical = Some(cap[1].to_string());
            }

            (error, warning, info, debug, critical)
        })
        .fold(
            || (vec![], vec![], vec![], vec![], vec![]),
            |mut acc, (e, w, i, d, c)| {
                if let Some(e) = e { acc.0.push(e); }
                if let Some(w) = w { acc.1.push(w); }
                if let Some(i) = i { acc.2.push(i); }
                if let Some(d) = d { acc.3.push(d); }
                if let Some(c) = c { acc.4.push(c); }
                acc
            },
        )
        .reduce(
            || (vec![], vec![], vec![], vec![], vec![]),
            |mut acc, item| {
                acc.0.extend(item.0);
                acc.1.extend(item.1);
                acc.2.extend(item.2);
                acc.3.extend(item.3);
                acc.4.extend(item.4);
                acc
            },
        );

    LogOutput { file: "logfile".to_string(), errors, warnings, infos, debugs, criticals }
}


#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("logfix")
        .version("1.0")
        .about("A CLI tool for analyzing and fixing error logs")
        .arg(Arg::new("json").long("json").help("Output log content in JSON format").action(ArgAction::SetTrue))
        .arg(Arg::new("diff").long("diff").help("Show differences between original and fixed logs")
        .num_args(0..=1).value_name("FIXED_FILE")
        .action(ArgAction::SetTrue)) // ← これを追加！
        .arg(Arg::new("fix").long("fix").help("Automatically fix errors in the log").action(ArgAction::SetTrue))
        .arg(Arg::new("fixed")
        .long("fixed")
        .help("Path to fixed log file")
        .num_args(1))
        .arg(Arg::new("file").help("Path to the error log file").required(true))
        .arg(Arg::new("ai-mode")
        .long("ai-mode")
        .help("Use advanced AI mode for full log analysis")
        .value_parser(["simple", "full"]))
        .get_matches();

    let file_path = matches.get_one::<String>("file").unwrap(); 
    let log_content = fs::read_to_string(file_path)?; // ← 追加
    let log_output = process_logs_by_level(&log_content);
    if let Some(ai_mode) = matches.get_one::<String>("ai-mode") {
        if ai_mode == "full" {
            println!("🚀 Running in **FULL AI Mode**: Deep log analysis with improvements...");
            let fix_suggestion = get_fix_suggestion(&log_content, "full").await.expect("AI Fix failed");
            println!("📝 AI Analysis:\n{}", fix_suggestion);
        }
    }
    if matches.get_flag("diff") {
            let original_file = matches.get_one::<String>("file").unwrap(); // ここで取得
            let fixed_file = matches.get_one::<String>("fixed_file").unwrap(); // 修正後のファイルを取得
        
            let original_content = fs::read_to_string(original_file)?;
            let fixed_content = fs::read_to_string(fixed_file)?;
        
            println!("✅ DEBUG: show_diff() を実行します");
            show_diff(&original_content, &fixed_content);
            println!("✅ DEBUG: show_diff() の処理が終了しました");
    }
    if matches.get_flag("fix") {
        println!("🔧 Fixing errors in log file: {}", file_path);
        for line in log_content.lines() {
            println!("{}", colorize_log(line)); // 🔥 色付きで出力！
        }
        let fix_suggestion = get_fix_suggestion(&log_content, "simple").await?;
        println!("📝 Fixed Log:\n{}", fix_suggestion);

    }
    let log_output_text = format!(
    "{}\n{}\n{}\n{}\n{}",
    log_output.errors.join("\n"),
    log_output.warnings.join("\n"),
    log_output.infos.join("\n"),
    log_output.debugs.join("\n"),
    log_output.criticals.join("\n")
);
    if matches.get_flag("json") {
        println!("{}", serde_json::to_string_pretty(&log_output).unwrap());
    } else if log_output_text.contains("impossible to provide any corrected code") {
        println!("❌ No source code found in the log.");
        println!("💡 Please provide a log containing actual code for `logfix --fix` to generate corrections.");
    } else {
        println!("Processed log output.");
    }
    
    Ok(())
}

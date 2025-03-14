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

async fn get_fix_suggestion(log: &str, ai_mode: &str) -> Result<String> {
    let api_url = "https://api.openai.com/v1/chat/completions";
    let client = Client::new();
    let prompt = format!(
        "Rewrite the following log with all errors fixed.
         Convert errors into successful messages where possible.
         Do not provide explanations, only return the corrected log content:\n{}",
        log
    );    
    let response = client.post(api_url)
        .header("Authorization", "Bearer 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")
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{ "role": "system", "content": "You are an expert log analyzer." },
                         { "role": "user", "content": prompt }],
            "temperature": 0.7
        }))
        .send()
        .await
        .context("Failed to send request to OpenAI API")?
        .json::<serde_json::Value>()
        .await
        .context("Failed to parse response from OpenAI API")?;
    
    if let Some(choice) = response["choices"].get(0) {
        if let Some(text) = choice["message"]["content"].as_str() {
            return Ok(text.to_string());
        }
    }
    
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
        .arg(Arg::new("diff").long("diff").help("Show differences between original and fixed logs").num_args(0..=1).value_name("FIXED_FILE"))
        .arg(Arg::new("fix").long("fix").help("Automatically fix errors in the log").action(ArgAction::SetTrue))
        .arg(Arg::new("fixed")
        .long("fixed")
        .help("Path to fixed log file")
        .num_args(1))
        .arg(Arg::new("file").help("Path to the error log file").required(true))
        .get_matches();

    let file_path = matches.get_one::<String>("file").unwrap(); 
    let log_content = fs::read_to_string(file_path)?; // ← 追加
let log_output = process_logs_by_level(&log_content);
        .version("1.0")
        .about("A CLI tool for analyzing and fixing error logs")
        .arg(Arg::new("json").long("json").help("Output log content in JSON format").action(ArgAction::SetTrue))
        .arg(Arg::new("diff").long("diff").help("Show differences between original and fixed logs").num_args(0..=1).value_name("FIXED_FILE"))
        .arg(Arg::new("fix").long("fix").help("Automatically fix errors in the log").action(ArgAction::SetTrue))
        .arg(Arg::new("fixed")
        .long("fixed")
        .help("Path to fixed log file")
        .num_args(1))
        .arg(Arg::new("file").help("Path to the error log file").required(true))
        .get_matches();
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
    
        let fix_suggestion = get_fix_suggestion(&log_content, "simple").await?;
        println!("📝 Fixed Log:\n{}", fix_suggestion);

    }
    
    if matches.get_flag("json") {
        println!("{}", serde_json::to_string_pretty(&log_output).unwrap());
    } else {
        println!("Processed log output.");
    }

    
    Ok(())
}

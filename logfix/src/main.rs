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

#[derive(Serialize, Deserialize)]
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
    let mut colored_log = log.to_string();

    if log.contains("ERROR") {
        colored_log = colored_log.red().to_string();
    }
    if log.contains("WARNING") {
        colored_log = colored_log.yellow().to_string();
    }
    if log.contains("INFO") {
        colored_log = colored_log.blue().to_string();
    }
    if log.contains("DEBUG") {
        colored_log = colored_log.green().to_string();
    }
    if log.contains("CRITICAL") {
        colored_log = colored_log.red().bold().to_string();
    }

    colored_log
}


pub fn optimize_log_data(log_data: &str, max_tokens: usize) -> Result<String> {
    let bpe = cl100k_base()?;
    let tokens = bpe.encode_with_special_tokens(log_data);

    if tokens.len() <= max_tokens {
        return Ok(log_data.to_string());
    }

    let optimized_log = bpe.decode(tokens[..max_tokens].to_vec())?;
    Ok(optimized_log)
}

async fn get_fix_suggestion(log: &str, ai_mode: &str) -> Result<String> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner}  {msg}").unwrap());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message("Analyzing log with AI...");

    let optimized_log = optimize_log_data(log, 4096)?;

    let api_url = "https://api.openai.com/v1/chat/completions";
    let client = Client::new();
    let prompt = format!(
        "Analyze the following log file and generate fixes. AI Mode: {}\n\n```log\n{}\n```",
        ai_mode, optimized_log
    );

    let response = client.post(api_url)
        .header("Authorization", format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{ "role": "system", "content": "You are an expert log analyzer." },
                         { "role": "user", "content": prompt }],
            "temperature": 0.7
        }))
        .send()
        .await?;

    pb.finish_with_message("✅ AI Analysis complete!");

    let response_json = response.json::<OpenAIResponse>().await?;
    if let Some(choice) = response_json.choices.get(0) {
        return Ok(choice.message.content.clone());
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
        "json" => serde_json::from_str(contents).unwrap_or_else(|_| vec![contents.to_string()]),
        "yaml" => serde_yaml::from_str(contents).unwrap_or_else(|_| vec![contents.to_string()]),
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
        .map(|line| (
            error_regex.captures(line).map(|cap| cap[1].to_string()),
            warning_regex.captures(line).map(|cap| cap[1].to_string()),
            info_regex.captures(line).map(|cap| cap[1].to_string()),
            debug_regex.captures(line).map(|cap| cap[1].to_string()),
            critical_regex.captures(line).map(|cap| cap[1].to_string()),
        ))
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

    LogOutput { errors, warnings, infos, debugs, criticals }
}

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("logfix")
        .version("1.0")
        .about("A CLI tool for analyzing and fixing error logs")
        .arg(Arg::new("json").long("json").help("Output log content in JSON format").action(ArgAction::SetTrue))
        .arg(
            Arg::new("diff")
                .long("diff")
                .help("Show differences between original and fixed logs")
                .num_args(2)
                .value_names(["ORIGINAL_FILE", "FIXED_FILE"])
        )        
        .arg(Arg::new("fix").long("fix").help("Automatically fix errors in the log").action(ArgAction::SetTrue))
        .arg(Arg::new("fixed")
            .long("fixed")
            .help("Path to fixed log file")
            .num_args(1))
        .arg(Arg::new("file")
            .help("Path to the error log file")
            .required(false))
        .arg(Arg::new("ai-mode")
            .long("ai-mode")
            .help("Use advanced AI mode for full log analysis")
            .value_parser(["simple", "full"]))
        .get_matches();

    // 🔹 `file` オプションがある場合のみ処理
    if let Some(file_path) = matches.get_one::<String>("file") {
        let log_content = fs::read_to_string(file_path)
            .expect("❌ Failed to read log file.");
        
        let log_output = process_logs_by_level(&log_content);

        // 🧠 AIモードの処理
        if let Some(ai_mode) = matches.get_one::<String>("ai-mode") {
            if ai_mode == "full" {
                println!("🚀 Running in **FULL AI Mode**: Deep log analysis with improvements...");
                let fix_suggestion = get_fix_suggestion(&log_content, "full").await?;
                println!("📝 AI Analysis:\n{}", fix_suggestion);
            }
        }

        // 🛠 `--fix` の処理
        if matches.get_flag("fix") {
            println!("🔧 Fixing errors in log file: {}", file_path);
            for line in log_content.lines() {
                println!("{}", colorize_log(line));
            }
            let fix_suggestion = get_fix_suggestion(&log_content, "simple").await?;
            println!("📝 Fixed Log:\n{}", fix_suggestion);
        }

        // 📝 ログ出力処理
        let log_output_text = format!(
            "{}\n{}\n{}\n{}\n{}",
            log_output.errors.join("\n"),
            log_output.warnings.join("\n"),
            log_output.infos.join("\n"),
            log_output.debugs.join("\n"),
            log_output.criticals.join("\n")
        );

        if matches.get_flag("json") {
            println!("{}", serde_json::to_string_pretty(&log_output)?);
        } else if log_output_text.contains("impossible to provide any corrected code") {
            println!("❌ No source code found in the log.");
            println!("💡 Please provide a log containing actual code for `logfix --fix` to generate corrections.");
        } else {
            println!("Processed log output.");
        }
    }

    // 📄 `--diff` の処理
    
    if let Some(mut files) = matches.get_many::<String>("diff") {
        let original_file = files.next().expect("Missing original file");
        let fixed_file = files.next().expect("Missing fixed file");

        let original_content = fs::read_to_string(original_file)
            .with_context(|| format!("Failed to read original file: {}", original_file))?;
        let fixed_content = fs::read_to_string(fixed_file)
            .with_context(|| format!("Failed to read fixed file: {}", fixed_file))?;

        println!("✅ DEBUG: show_diff() を実行します");
        show_diff(&original_content, &fixed_content);
        println!("✅ DEBUG: show_diff() の処理が終了しました");
    }

    Ok(())
}
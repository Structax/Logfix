use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use clap::{Arg, Command, ArgAction};
use serde::{Serialize, Deserialize};
use serde_json;
use serde_yaml;
use toml;
use reqwest;
use tokio;
use regex::Regex;
use similar::{TextDiff, Algorithm};

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
        "toml" => contents.lines().map(String::from).collect(),
        _ => contents.lines().map(String::from).collect(),
    }
}

fn process_logs_by_level(contents: &str) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let error_regex = Regex::new(r"(?i)error[: ](.*)").unwrap();
    let warning_regex = Regex::new(r"(?i)warning[: ](.*)").unwrap();
    let info_regex = Regex::new(r"(?i)info[: ](.*)").unwrap();
    let debug_regex = Regex::new(r"(?i)debug[: ](.*)").unwrap();
    let critical_regex = Regex::new(r"(?i)critical[: ](.*)").unwrap();
    
    let lines = parse_log(contents);
    let (errors, warnings, infos, debugs, criticals): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = lines
        .par_iter()
        .map(|line| {
            if let Some(cap) = error_regex.captures(line) {
                (Some(cap[1].to_string()), None, None, None, None)
            } else if let Some(cap) = warning_regex.captures(line) {
                (None, Some(cap[1].to_string()), None, None, None)
            } else if let Some(cap) = info_regex.captures(line) {
                (None, None, Some(cap[1].to_string()), None, None)
            } else if let Some(cap) = debug_regex.captures(line) {
                (None, None, None, Some(cap[1].to_string()), None)
            } else if let Some(cap) = critical_regex.captures(line) {
                (None, None, None, None, Some(cap[1].to_string()))
            } else {
                (None, None, None, None, None)
            }
        })
        .fold(|| (vec![], vec![], vec![], vec![], vec![]), |mut acc, item| {
            if let Some(e) = item.0 { acc.0.push(e); }
            if let Some(w) = item.1 { acc.1.push(w); }
            if let Some(i) = item.2 { acc.2.push(i); }
            if let Some(d) = item.3 { acc.3.push(d); }
            if let Some(c) = item.4 { acc.4.push(c); }
            acc
        })
        .reduce(|| (vec![], vec![], vec![], vec![], vec![]), |mut acc, item| {
            acc.0.extend(item.0);
            acc.1.extend(item.1);
            acc.2.extend(item.2);
            acc.3.extend(item.3);
            acc.4.extend(item.4);
            acc
        });
    (errors, warnings, infos, debugs, criticals)
}

fn show_diff(original: &str, modified: &str) {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_lines(original, modified);
    
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Delete => print!("-{}", change),
            similar::ChangeTag::Insert => print!("+{}", change),
            similar::ChangeTag::Equal => print!(" {}", change),
        }
    }
}

#[tokio::main]
async fn main() {
    let matches = Command::new("logfix")
        .version("1.0")
        .author("Your Name <your_email@example.com>")
        .about("A CLI tool for analyzing and fixing error logs")
        .arg(Arg::new("file")
            .help("Path to the error log file")
            .required(true)
            .index(1))
        .arg(Arg::new("json")
            .long("json")
            .help("Output log content in JSON format")
            .action(ArgAction::SetTrue))
        .arg(Arg::new("diff")
            .long("diff")
            .help("Show differences between original and fixed logs")
            .action(ArgAction::SetTrue))
        .get_matches();

    if let Some(file_path) = matches.get_one::<String>("file") {
        let path = PathBuf::from(file_path);
        match fs::read_to_string(&path) {
            Ok(contents) => {
                let (errors, warnings, infos, debugs, criticals) = process_logs_by_level(&contents);
                if matches.get_flag("json") {
                    let output = LogOutput {
                        file: file_path.clone(),
                        errors,
                        warnings,
                        infos,
                        debugs,
                        criticals,
                    };
                    println!("{}", serde_json::to_string_pretty(&output).unwrap());
                } else {
                    println!("Processed log output.");
                }
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
}

use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use std::net::TcpListener;

fn get_available_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(8000)
}

#[tauri::command]
fn get_api_port(port: tauri::State<u16>) -> u16 {
    *port
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  let port = get_available_port();

  tauri::Builder::default()
    .manage(port)
    .invoke_handler(tauri::generate_handler![get_api_port])
    .plugin(tauri_plugin_shell::init())
    .setup(move |app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      
      let sidecar_command = app.shell().sidecar("firefly_api").unwrap().env("FIREFLY_API_PORT", port.to_string());
      let (mut rx, mut _child) = sidecar_command.spawn().expect("Failed to spawn sidecar");
      
      tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
          if let CommandEvent::Stdout(line) = event {
            println!("Sidecar: {:?}", String::from_utf8(line).unwrap_or_default());
          }
        }
      });
      
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

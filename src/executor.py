# src/executor.py
import subprocess
import webbrowser
import platform
import os

def execute_action(intent: str, slots: dict):
    """
    Routes the predicted intent to the correct execution function.
    """
    print(f"--- Executing action for intent: {intent} with slots: {slots} ---")

    # A simple router to call the correct function based on the intent
    if intent == "os.app.open":
        app_name = slots.get("app_name")
        if app_name:
            open_app(app_name)
        else:
            print("Error: app_name not found in slots.")

    elif intent == "os.app.close":
        app_name = slots.get("app_name")
        if app_name:
            close_app(app_name)
        else:
            print("Error: app_name not found in slots.")

    elif intent == "browser.open":
        url = slots.get("url")
        if url:
            open_url(url)
        else:
            print("Error: url not found in slots.")
    
    # Add more intent handlers here in the future
    # elif intent == "browser.search":
    #     ...

    else:
        print(f"Execution for intent '{intent}' is not implemented yet.")


def open_app(app_name: str):
    """
    Tries to open an application locally. If it fails, it attempts to
    open a corresponding website as a fallback.
    """
    os_name = platform.system()
    print(f"Attempting to open '{app_name}' on {os_name}...")
    
    official_app_name = app_name # Default to the name we were given

    try:
        if os_name == "Darwin":  # macOS
            app_name_map = {
                "chrome": "Google Chrome", "google chrome": "Google Chrome", "brave": "Brave Browser",
                "brave browser": "Brave Browser", "firefox": "Firefox", "safari": "Safari",
                "vscode": "Visual Studio Code", "vs code": "Visual Studio Code", "terminal": "Terminal",
                "iterm": "iTerm", "notes": "Notes", "calendar": "Calendar", "mail": "Mail",
                "settings": "System Settings", "system settings": "System Settings",
                "preferences": "System Settings", "system preferences": "System Settings",
                "spotify": "Spotify", "netflix": "Netflix", "apple tv": "TV", "tv": "TV", "music": "Music",
            }
            official_app_name = app_name_map.get(app_name.lower(), app_name)
            
            print(f"Resolved app name to '{official_app_name}'")
            subprocess.run(["open", "-a", official_app_name], check=True)
            print(f"Successfully launched {official_app_name}.")

        elif os_name == "Windows":
            # Add a similar map for Windows here if needed
            os.startfile(f"{app_name}.exe")
            print(f"Successfully launched {app_name}.")
        
        elif os_name == "Linux":
            # Add a similar map for Linux here if needed
            subprocess.run([app_name.lower()], check=True)
            print(f"Successfully launched {app_name}.")
        else:
            print(f"Unsupported operating system: {os_name}")

    except Exception as e:
        # This is our new fallback logic!
        print(f"Error opening local application '{official_app_name}': {e}")
        print(f"--- Fallback: Attempting to open '{app_name}' as a website. ---")
        
        # --- MODIFICATION START ---
        # Create a mapping for common services to their correct websites.
        website_map = {
            "netflix": "netflix.com",
            "amazon prime": "primevideo.com",
            "prime video": "primevideo.com",
            "youtube": "youtube.com",
            "gmail": "gmail.com",
            "google drive": "drive.google.com",
        }
        
        # Check if the app name is in our special map.
        url_guess = website_map.get(app_name.lower())
        
        # If not found in the map, use the original guessing logic.
        if not url_guess:
            url_guess = f"{app_name.lower().replace(' ', '')}.com"
        
        open_url(url_guess)
        # --- MODIFICATION END ---


def close_app(app_name: str):
    """
    Closes an application based on the operating system.
    """
    os_name = platform.system()
    print(f"Attempting to close '{app_name}' on {os_name}...")

    try:
        if os_name == "Darwin":  # macOS
            app_name_map = {
                "chrome": "Google Chrome", "google chrome": "Google Chrome", "brave": "Brave Browser",
                "brave browser": "Brave Browser", "firefox": "Firefox", "safari": "Safari",
                "vscode": "Visual Studio Code", "vs code": "Visual Studio Code", "terminal": "Terminal",
                "iterm": "iTerm", "notes": "Notes", "calendar": "Calendar", "mail": "Mail",
                "settings": "System Settings", "system settings": "System Settings",
                "preferences": "System Settings", "system preferences": "System Settings",
                "spotify": "Spotify", "netflix": "Netflix", "apple tv": "TV", "tv": "TV", "music": "Music",
            }
            official_app_name = app_name_map.get(app_name.lower(), app_name)
            
            print(f"Resolved app name to '{official_app_name}'")
            command = f'osascript -e \'quit app "{official_app_name}"\''
            subprocess.run(command, shell=True, check=True)
            print(f"Successfully closed {official_app_name}.")
        elif os_name == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", f"{app_name}.exe"], check=True)
            print(f"Successfully closed {app_name}.")
        elif os_name == "Linux":
            subprocess.run(["pkill", app_name.lower()], check=True)
            print(f"Successfully closed {app_name}.")
        else:
            print(f"Unsupported operating system: {os_name}")
    except Exception as e:
        print(f"Error closing application '{app_name}': {e}")
        print("Please ensure the application is running and the name is correct.")


def open_url(url: str):
    """
    Opens a URL in the default web browser.
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"Opening URL: {url}")
        webbrowser.open(url)
        print("URL opened successfully in default browser.")
    except Exception as e:
        print(f"Error opening URL '{url}': {e}")

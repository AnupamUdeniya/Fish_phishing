# CMPE209FinalProject - Phising extension using LLM

Requirement
   - NodeJS version v20.0.0 or greater
   - Python3
- Build the backend:
  ```sh
        cd backend
        python3 -m venv venv
        # Depending on if you're using macOS or Windows, use the following commands:
           - MacOS: source venv/bin/activate
           - Windows: venv\Scripts\activate
        pip install -r requirements.txt
        python app.py
    ```

- Build the extension:
    ```sh
        cd frontend/phishing-extension
        npm install
        npm run build
    ```

- Install the extension locally:
    - Step 1: Access the extension.
        - In Google Chrome browser tab bar, type chrome://extensions/
    - Step 2: Turn on developer mode:
        - Toggle on the developer mode located at the upper right of the extension window. This allow us to load the extension for testing purpose
    - Step 3: Load the unpacked extension.
        - After the developer mode is turned on, browser will show you the button upload extension. Simply click to the Load Unpacked button and navigate to the extension folder in the project to upload the extension.

# Final Demo/Presentation Video
https://drive.google.com/file/d/1da0vrgniRjeGQX0DhJtp49EqYFVWwu5Z/view?usp=drive_link

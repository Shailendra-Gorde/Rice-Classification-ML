# Rice Classification ML - Setup Guide

## Prerequisites
- Python 3.11+ 
  ```cmd
  python --version
  ```

- Node.js 14+ 
  ```cmd
  node --version
  ```

- npm (comes with Node.js)

## Step-by-Step Setup Instructions

**Note:** All commands should be run in Command Prompt (cmd), not PowerShell.

Example prompt: `C:\Users\JSE40\OneDrive\Documents\project\Rice-Classification-ML>`

**If you already have a venv folder and want to recreate it:**
```cmd
rmdir /s /q venv
```



### BACKEND SETUP (Python/Flask)

#### Step 1: Navigate to Project Directory
```cmd
cd C:\Users\JSE40\OneDrive\Documents\project\Rice-Classification-ML
```

#### Step 2: Create Virtual Environment
```cmd
python -m venv venv
```

#### Step 3: Activate Virtual Environment

**For Command Prompt (cmd):**
```cmd
venv\Scripts\activate
```

**You'll know it's activated when you see `(venv)` at the start of your command prompt.**

#### Step 4: Install Python Dependencies
```cmd
pip install -r requirements.txt
```

**If the above command fails, use this instead:**
```cmd
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

**Note:** If you encounter authentication errors with a private repository, use the `--index-url` flag to use the public PyPI.

#### Step 5: Run the Backend Server
```cmd
python api_server.py
```

The backend will start on **http://localhost:5001**

---

### FRONTEND SETUP (React)

#### Step 1: Navigate to Frontend Directory

```cmd
cd rice-ml-dashboard
```

**After this, your prompt will look like:**
```
C:\Users\JSE40\OneDrive\Documents\project\Rice-Classification-ML\rice-ml-dashboard>
```

#### Step 2: Install Node.js Dependencies

```cmd
npm install
```

This will install all React dependencies (react, react-dom, Material-UI, etc.)

#### Step 3: Start the Frontend Development Server

```cmd
npm start
```

The frontend will start on **http://localhost:3000** (default React port)

---

## Running Both Servers

### Option 1: Two Separate Terminal Windows

**Terminal 1 - Backend:**
```cmd
cd C:\Users\JSE40\OneDrive\Documents\project\Rice-Classification-ML
venv\Scripts\activate
python api_server.py
```

**Terminal 2 - Frontend:**
```cmd
cd C:\Users\JSE40\OneDrive\Documents\project\Rice-Classification-ML\rice-ml-dashboard
npm start
```


## Troubleshooting

### Port Already in Use
- Backend uses port 5001 - if it's busy, you can change it in `api_server.py` 
- Frontend uses port 3000 - React will ask if you want to use a different port

### Dependencies
- Backend: Make sure virtual environment is activated before installing
- Frontend: Make sure you're in `rice-ml-dashboard` directory when running `npm install`

---

## Accessing the Application

- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:5001
- **API Health Check:** http://localhost:5001/api/health

---

## Notes

- Keep both servers running while using the application
- The backend must be running for the frontend to work properly
- The frontend communicates with the backend API for predictions


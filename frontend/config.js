/* Empty API base means "same origin as the page" (relative fetches) --
   correct both for local dev (`uvicorn backend.main:app`) and for the
   deployed app, since backend/main.py serves the front-end and the API
   from the same process/origin. */
window.DEEPSYNC_API_BASE = "";

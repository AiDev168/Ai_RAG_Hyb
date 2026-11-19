"""Entry point:  streamlit run app.py

Importing `aura_rag` first installs the Offline/Online network guard before any
networking or model library gets imported.
"""
import aura_rag  # noqa: F401  (must stay first)
from aura_rag.ui.main_view import main

if __name__ == "__main__":
    main()

"""RetinaTrace AI Gradio application package.

Modules:
    main      - builds the Gradio Blocks interface and launches it (entry point)
    analysis  - fundus analysis handler, image quality checks, longitudinal comparison
    chatbot   - clinical knowledge base and the router / knowledge / governance chat agents
    reports   - JSON and PDF reports, EHR session note, referral ticket
    triage    - illustrative multimodal risk simulator (not validated)
    ui_style  - CSS, HTML and JavaScript blocks used by the interface

The root-level app.py imports app.main and launches it.
"""

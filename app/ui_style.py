"""Interface styling: CSS, head JavaScript, theme, logo and the fixed sidebar/header HTML."""

import os

import gradio as gr

from core.config import PROJECT_ROOT


CUSTOM_CSS = """
<style>
/* Medical Dashboard Base Styling */
html, body {
    overflow-x: hidden !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #edf3f8 !important;
    color: #0f172a !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}
.dark, html.dark, body.dark {
    background-color: #0b1120 !important;
    color: #f8fafc !important;
}

#root,
#app,
gradio-app {
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
    max-width: 100% !important;
    width: 100% !important;
    background-color: transparent !important;
}

.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #edf3f8 !important;
    transition: background-color 0.3s ease, color 0.3s ease;
    box-sizing: border-box !important;
    margin: 0 auto !important;
    padding: 0 12px 12px !important;
}

html.dark .gradio-container,
body.dark .gradio-container,
gradio-app.dark .gradio-container,
.dark .gradio-container {
    background-color: #0b1120 !important;
    color: #f8fafc !important;
}

.row,
.gr-row {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 12px !important;
    width: 100% !important;
    max-width: 100% !important;
}

.column,
.gr-column {
    width: 100% !important;
    max-width: 100% !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
}

.row > .column:first-child,
.gradio-container .gr-row > div:first-child,
.row > .column:last-child,
.gradio-container .gr-row > div:last-child {
    width: 100% !important;
    max-width: 100% !important;
    flex: 0 0 100% !important;
    min-width: 0 !important;
}

@media (min-width: 901px) {
    #root,
    #app,
    gradio-app {
        margin-left: 240px !important;
        width: calc(100% - 240px) !important;
        max-width: calc(100% - 240px) !important;
    }
}

.gradio-container > .main,
.gradio-container > .main > .wrap,
#component-0 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    gap: 0 !important;
}

/* Header Telemetry Styling */
.telemetry-bar {
    display: flex;
    gap: 12px;
    background: #ffffff;
    color: #0f172a;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 12px;
    margin-bottom: 16px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.telemetry-bar strong {
    color: #0f172a !important;
}
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f1f5f9;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    font-weight: 500;
    color: #334155 !important;
}
.dark .telemetry-bar {
    background: #0f172a;
    border: 1px solid #1e293b;
    box-shadow: none;
}
.dark .telemetry-bar strong {
    color: #f8fafc !important;
}
.dark .telemetry-badge {
    background: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc !important;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Base Card Styling */
.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
    color: #1e293b;
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}
.dark .card {
    background: #1e293b;
    border: 1px solid #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    color: #f1f5f9;
}
.clinical-document-box textarea {
    min-height: 260px !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    white-space: pre-wrap !important;
    background: #f8fafc !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}
.dark .clinical-document-box textarea {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
}
.icon {
    font-size: 24px;
}

/* Alert, Success, Warning Cards */
.success-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.dark .success-card {
    background: #064e3b;
    border: 1px solid #059669;
}
.alert-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.dark .alert-card {
    background: #450a0a;
    border: 1px solid #dc2626;
}
.warning-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.dark .warning-card {
    background: #451a03;
    border: 1px solid #d97706;
    color: #fef3c7;
}

/* Hero Diagnostic Card */
.hero-card {
    background: #ffffff;
}
.dark .hero-card {
    background: #1e293b;
}
.hero-stage-title {
    color: #0f172a;
}
.dark .hero-stage-title {
    color: #f8fafc;
}
.hero-subtext, .hero-sublabel {
    color: #64748b;
}
.dark .hero-subtext, .dark .hero-sublabel {
    color: #94a3b8;
}

/* Stage Badge Dark Mode Overrides */
.dark .stage-badge-0 { background: rgba(16, 185, 129, 0.2) !important; color: #34d399 !important; }
.dark .stage-badge-1 { background: rgba(2, 132, 199, 0.2) !important; color: #38bdf8 !important; }
.dark .stage-badge-2 { background: rgba(217, 119, 6, 0.2) !important; color: #fbbf24 !important; }
.dark .stage-badge-3 { background: rgba(234, 88, 12, 0.2) !important; color: #fb923c !important; }
.dark .stage-badge-4 { background: rgba(225, 29, 72, 0.2) !important; color: #f87171 !important; }

/* Protocol Section */
.protocol-title {
    color: #0f172a;
    border-bottom: 1px solid #e2e8f0;
}
.dark .protocol-title {
    color: #f8fafc;
    border-bottom: 1px solid #334155;
}
.protocol-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.dark .protocol-box {
    background: #0f172a;
    border: 1px solid #334155;
}
.protocol-label {
    color: #64748b;
}
.dark .protocol-label {
    color: #94a3b8;
}
.protocol-val {
    color: #0f172a;
}
.dark .protocol-val {
    color: #f8fafc;
}
.action-box {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
}
.dark .action-box {
    background: #0f172a;
    border: 1px solid #334155;
}
.action-label {
    color: #475569;
}
.dark .action-label {
    color: #94a3b8;
}
.action-val {
    color: #1e293b;
}
.dark .action-val {
    color: #f1f5f9;
}
.disclaimer-text {
    color: #94a3b8;
}
.dark .disclaimer-text {
    color: #64748b;
}

/* Header Text */
.header-title {
    color: #0f172a;
}
.dark .header-title {
    color: #f8fafc;
}
.header-subtitle {
    color: #475569;
}
.dark .header-subtitle {
    color: #94a3b8;
}

/* Theme Toggle Button */
.theme-toggle-btn {
    border-radius: 8px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    background: #f0fdfa !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.theme-toggle-btn:hover {
    background: #f1f5f9 !important;
    transform: translateY(-1px);
}
.dark .theme-toggle-btn {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    box-shadow: none !important;
}
.dark .theme-toggle-btn:hover {
    background: #334155 !important;
}

/* Button & Tool Enhancements */
.action-btn {
    background: linear-gradient(135deg, #0d9488, #0284c7) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
.action-btn:hover {
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
}

.risk-card-header {
    flex-wrap: wrap;
    gap: 8px;
}
.risk-ladder > div {
    min-width: 0;
}

/* Keep the first viewport focused on the primary action and prediction. */
.gradio-container .gr-accordion {
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
.gradio-container .gr-accordion > .label-wrap {
    padding: 8px 12px !important;
}
.gradio-container .tabs {
    margin-top: 4px !important;
}
.gradio-container .tabitem {
    padding-top: 6px !important;
}

/* ── Global padding compression ── */
/* Reduce the default Gradio page-level top/bottom padding */
.gradio-container {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
/* Tighten row & column gaps */
.gradio-container .gap {
    gap: 8px !important;
}
/* Shrink individual form-block vertical margins */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap > * {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* Image upload widget label/label-wrap */
.gradio-container .block label,
.gradio-container .block .label-wrap {
    margin-bottom: 2px !important;
}
/* Reduce space between the uploader and the Run button row */
.gradio-container .row {
    gap: 6px !important;
}
/* Card spacing */
.card {
    margin-bottom: 8px !important;
}

/* ── Aggressive Gradio internal spacing overrides ── */
/* Remove top padding on the main app wrapper */
.gradio-container > .main,
.gradio-container > .main > .wrap {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
    gap: 8px !important;
}
/* Shrink Gradio svelte block wrapper padding */
.svelte-1gfkn6j,
[class*="wrap "] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
/* Tighten Gradio built-in form block top/bottom padding */
.form > div,
.form > .block {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
/* Force image block not to add extra margin */
.gradio-container .image-frame {
    margin: 0 !important;
}
/* Shrink button groups internal margin */
.gradio-container button + button {
    margin-left: 4px !important;
}
/* Tighten inter-element gap inside each column */
.gradio-container > .main > .wrap > .contain > * + * {
    margin-top: 6px !important;
}

/* Keep dense result cards usable on phones and small tablet widths. */
@media (max-width: 700px) {
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .header-title {
        font-size: 21px !important;
        line-height: 1.2 !important;
    }

    .telemetry-bar {
        align-items: flex-start;
        flex-direction: column;
    }

    .telemetry-bar > div:last-child {
        width: 100%;
    }

    .telemetry-badge {
        font-size: 11px;
        padding: 4px 7px;
    }

    .hero-stage-title {
        font-size: 21px !important;
    }

    .gallery {
        min-width: 0 !important;
    }

    .chatbot {
        height: 320px !important;
    }

    .responsive-grid-two,
    .responsive-grid-three {
        grid-template-columns: 1fr !important;
    }

    .risk-card {
        padding: 10px !important;
    }

    .risk-card-header > div:first-child {
        max-width: 100%;
        line-height: 1.35;
    }

    .risk-ladder {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px !important;
    }

    .risk-ladder > div {
        min-width: 0 !important;
    }

    .gradio-container .gr-accordion > .label-wrap {
        padding: 8px 10px !important;
    }
}

/* Stepper Component (Medios / Mobile Workflow) */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f0fdfa;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 6px 12px;
    margin-bottom: 8px;
    color: #065f46 !important;
    flex-wrap: wrap;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.stepper-item {
    display: flex;
    align-items: center;
    gap: 10px;
}
.stepper-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #0d9488;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
}
.stepper-title {
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #065f46 !important;
}
.stepper-sub {
    font-size: 10px;
    color: #047857 !important;
    opacity: 0.85;
}
.stepper-arrow {
    color: #059669 !important;
    font-weight: 800;
    font-size: 14px;
}
.dark .stepper-container {
    background: #022c22;
    border-color: #134e4a;
    box-shadow: none;
}
.dark .stepper-title {
    color: #ccfbf1 !important;
}
.dark .stepper-sub {
    color: #99f6e4 !important;
}
.dark .stepper-arrow {
    color: #2dd4bf !important;
}

/* Medios Patient EHR Card */
.patient-id-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 8px 10px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dark .patient-id-card {
    background: #1e293b;
    border-color: #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.badge-quota {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    font-size: 10px;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.dark .badge-quota {
    background: #064e3b;
    color: #a7f3d0;
    border-color: #059669;
}
.patient-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #f1f5f9;
    color: #334155;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #e2e8f0;
}
.dark .patient-chip {
    background: #1e293b;
    color: #cbd5e1;
    border-color: #334155;
}

/* Medios SaMD Regulatory Card */
.medios-disclaimer-card {
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 14px;
    color: #0f766e;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.medios-disclaimer-card .disclaimer-title {
    color: #0d9488 !important;
}
.medios-disclaimer-card .disclaimer-body {
    color: #134e4a !important;
}
.dark .medios-disclaimer-card {
    background: #022c22;
    border-color: #134e4a;
    color: #ccfbf1;
    box-shadow: none;
}
.dark .medios-disclaimer-card .disclaimer-title {
    color: #2dd4bf !important;
}
.dark .medios-disclaimer-card .disclaimer-body {
    color: #ccfbf1 !important;
}

/* Gradio Tab Bar Styling */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 6px !important;
}
.dark .tabs > .tab-nav {
    border-bottom: 2px solid #1e293b !important;
}
.tabs > .tab-nav > button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    padding: 8px 14px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.15s ease !important;
    background: transparent !important;
}
.tabs > .tab-nav > button:hover {
    color: #0d9488 !important;
    background: #f1f5f9 !important;
}
.tabs > .tab-nav > button.selected {
    color: #0d9488 !important;
    border-bottom: 2px solid #0d9488 !important;
    font-weight: 700 !important;
}
.dark .tabs > .tab-nav > button {
    color: #94a3b8 !important;
}
.dark .tabs > .tab-nav > button:hover {
    color: #2dd4bf !important;
    background: #1e293b !important;
}
.dark .tabs > .tab-nav > button.selected {
    color: #2dd4bf !important;
    border-bottom: 2px solid #0d9488 !important;
}

/* ── Left Sidebar Navigation (ProvoHeal Inspired) ─────────────────── */
#rg-sidebar {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    height: 100vh !important;
    width: 240px !important;
    background: #0b1329 !important;
    border-right: 1px solid #1e293b !important;
    z-index: 99999 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.45) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    transition: width 0.25s ease !important;
}
#rg-sidebar * {
    box-sizing: border-box !important;
}
#rg-sidebar .rg-sb-logo {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: 20px 18px 16px !important;
    border-bottom: 1px solid #1e293b !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-logo-icon {
    width: 38px !important;
    height: 38px !important;
    background: rgba(46, 125, 50, 0.16) !important;
    border: 1px solid rgba(74, 222, 128, 0.32) !important;
    border-radius: 10px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 3px !important;
    flex-shrink: 0 !important;
    box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25) !important;
}
#rg-sidebar .rg-sb-logo-icon .rt-logo-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    filter: drop-shadow(0 0 5px rgba(74, 222, 128, 0.5)) brightness(1.2) contrast(1.1) !important;
}
#rg-sidebar .rg-sb-logo-text {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    line-height: 1.2 !important;
    letter-spacing: -0.2px !important;
}
#rg-sidebar .rg-sb-logo-sub {
    font-size: 10.5px !important;
    color: #86efac !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
.rt-logo-badge {
    width: 38px !important;
    height: 38px !important;
    border-radius: 10px !important;
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 4px !important;
    box-shadow: 0 2px 6px rgba(46, 125, 50, 0.12) !important;
    flex-shrink: 0 !important;
}
.rt-logo-badge .rt-logo-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}
.dark .rt-logo-badge {
    background: rgba(46, 125, 50, 0.22) !important;
    border-color: rgba(74, 222, 128, 0.35) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
}
.dark .rt-logo-badge .rt-logo-img {
    filter: drop-shadow(0 0 5px rgba(74, 222, 128, 0.45)) brightness(1.22) contrast(1.1) !important;
}
.dark .header-title {
    color: #f8fafc !important;
}
.dark .header-title span {
    color: #4ade80 !important;
}
.dark .rt-badge-pill {
    color: #4ade80 !important;
    background: rgba(74, 222, 128, 0.15) !important;
    border-color: rgba(74, 222, 128, 0.35) !important;
}
#rg-sidebar .rg-sb-search-wrap {
    padding: 12px 14px 6px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-search {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: #131f38 !important;
    border: 1px solid #223554 !important;
    border-radius: 8px !important;
    padding: 7px 10px !important;
    color: #94a3b8 !important;
}
#rg-sidebar .rg-sb-search input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #f1f5f9 !important;
    font-size: 11.5px !important;
    width: 100% !important;
}
#rg-sidebar .rg-sb-search input::placeholder {
    color: #64748b !important;
}
#rg-sidebar .rg-kbd {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 4px !important;
    font-size: 9.5px !important;
    font-weight: 700 !important;
    padding: 1px 5px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-section-label {
    font-size: 9.5px !important;
    font-weight: 800 !important;
    color: #475569 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 12px 18px 6px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-nav {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    padding: 0 10px !important;
    flex: 1 !important;
    overflow-y: auto !important;
}
#rg-sidebar .rg-sb-nav::-webkit-scrollbar { width: 3px; }
#rg-sidebar .rg-sb-nav::-webkit-scrollbar-thumb {
    background: #1e293b; border-radius: 3px;
}
#rg-sidebar .rg-sb-nav button {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    padding: 9px 12px !important;
    border: none !important;
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.1px !important;
    position: relative !important;
}
#rg-sidebar .rg-sb-nav button:hover {
    background: #162444 !important;
    color: #38bdf8 !important;
}
#rg-sidebar .rg-sb-nav button.rg-active {
    background: #132742 !important;
    color: #2dd4bf !important;
    border-left: 3px solid #0d9488 !important;
    font-weight: 700 !important;
}
#rg-sidebar .rg-sb-nav .rg-sb-icon {
    font-size: 15px !important;
    flex-shrink: 0 !important;
    width: 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#rg-sidebar .rg-badge {
    margin-left: auto !important;
    background: #0369a1 !important;
    color: #e0f2fe !important;
    font-size: 9.5px !important;
    font-weight: 700 !important;
    padding: 2px 6px !important;
    border-radius: 10px !important;
}
#rg-sidebar .rg-sb-telemetry {
    padding: 10px 14px !important;
    margin: 8px 10px !important;
    background: #0f1c34 !important;
    border: 1px solid #1e2e4f !important;
    border-radius: 8px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-status-dot {
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: #10b981 !important;
    box-shadow: 0 0 8px #10b981 !important;
    display: inline-block !important;
}
#rg-sidebar .rg-sb-footer {
    padding: 10px 12px 14px !important;
    border-top: 1px solid #1e293b !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-footer button {
    width: 100% !important;
    padding: 8px 12px !important;
    border: 1px solid #1e2e4f !important;
    background: #111e38 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 0.15s ease !important;
}
#rg-sidebar .rg-sb-footer button:hover {
    background: #192a4e !important;
    color: #f8fafc !important;
    border-color: #38bdf8 !important;
}
/* Floating chat pill bottom-right */
.rg-floating-chat-pill {
    position: fixed !important;
    right: 24px !important;
    bottom: 18px !important;
    z-index: 200000 !important;
    background: linear-gradient(135deg, #0d9488, #0284c7) !important;
    color: #ffffff !important;
    padding: 10px 18px !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.45) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    opacity: 1 !important;
    visibility: visible !important;
}
.rg-floating-chat-pill:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(13, 148, 136, 0.6) !important;
}

/* Compact assistant window opened from the floating pill */
.rg-chat-floating-panel {
    position: fixed !important;
    right: 24px !important;
    bottom: 66px !important;
    z-index: 100000 !important;
    width: min(560px, calc(100vw - 48px)) !important;
    height: auto !important;
    max-height: min(620px, calc(100vh - 110px)) !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    padding: 14px 16px 14px !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28) !important;
    animation: rg-chat-panel-in 0.18s ease-out !important;
    display: flex !important;
    flex-direction: column !important;
}
.dark .rg-chat-floating-panel {
    background: #102a43 !important;
    border-color: #334155 !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5) !important;
}
.rg-chat-floating-panel .rg-chat-close {
    position: absolute !important;
    top: 10px !important;
    right: 12px !important;
    z-index: 2 !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 50% !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    cursor: pointer !important;
}
.dark .rg-chat-floating-panel .rg-chat-close {
    border-color: #475569 !important;
    color: #cbd5e1 !important;
}
.rg-chat-floating-panel .rg-chat-close:hover {
    background: #e2e8f0 !important;
}
.dark .rg-chat-floating-panel .rg-chat-close:hover {
    background: #1e293b !important;
}
.rg-chat-floating-panel .rg-chat-specs {
    display: none !important;
}
.rg-chat-floating-panel [role="log"] {
    min-height: 0 !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    max-height: 170px !important;
}
.rg-chat-floating-panel .rg-chat-input-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 88px !important;
    align-items: center !important;
    gap: 8px !important;
    margin-top: 8px !important;
    width: 100% !important;
    min-width: 0 !important;
}
.rg-chat-floating-panel .rg-chat-input-row > div,
.rg-chat-floating-panel .rg-chat-input-row button {
    align-self: center !important;
}
.rg-chat-floating-panel .rg-chat-input-row > div {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
.rg-chat-floating-panel .rg-chat-input-row > div:last-child {
    width: 88px !important;
    max-width: 88px !important;
}
.rg-chat-floating-panel .rg-chat-input-row button {
    width: 88px !important;
    min-height: 46px !important;
    height: 46px !important;
}
.rg-chat-floating-panel .rg-chat-input-row textarea,
.rg-chat-floating-panel .rg-chat-input-row button {
    min-height: 46px !important;
    height: 46px !important;
    box-sizing: border-box !important;
}
.gradio-container .rg-chat-input-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 88px !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    min-width: 0 !important;
}
.gradio-container .rg-chat-input-row > div:first-child {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}
.gradio-container .rg-chat-input-row > div:last-child {
    width: 88px !important;
    max-width: 88px !important;
    min-width: 88px !important;
    flex: 0 0 88px !important;
}
.gradio-container .rg-chat-input-row button {
    width: 88px !important;
    min-width: 88px !important;
    min-height: 46px !important;
    height: 46px !important;
    align-self: center !important;
}
.rg-chat-input-row,
.rg-chat-input-row.row,
.rg-chat-input-row.gr-row,
.rg-chat-input-row > .row,
.rg-chat-input-row > .gr-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 88px !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    min-width: 0 !important;
}
.rg-chat-input-row > div,
.rg-chat-input-row > .row > div,
.rg-chat-input-row > .gr-row > div {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}
.rg-chat-input-row > div:last-child,
.rg-chat-input-row > .row > div:last-child,
.rg-chat-input-row > .gr-row > div:last-child {
    width: 88px !important;
    max-width: 88px !important;
    min-width: 88px !important;
    flex: 0 0 88px !important;
}
.rg-chat-input-row button,
.rg-chat-input-row > .row button,
.rg-chat-input-row > .gr-row button {
    width: 88px !important;
    min-width: 88px !important;
    min-height: 46px !important;
    height: 46px !important;
    align-self: center !important;
}
.rg-history-floating-panel {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    right: auto !important;
    bottom: auto !important;
    z-index: 100000 !important;
    width: min(560px, calc(100vw - 48px)) !important;
    max-height: min(620px, calc(100vh - 110px)) !important;
    overflow-y: auto !important;
    padding: 14px 16px !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28) !important;
    transform: translate(-50%, -50%) !important;
    animation: rg-history-panel-in 0.18s ease-out !important;
}
.dark .rg-history-floating-panel {
    background: #102a43 !important;
    border-color: #334155 !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5) !important;
}
.rg-history-floating-panel .rg-history-close {
    float: right !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 50% !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    cursor: pointer !important;
}
.dark .rg-history-floating-panel .rg-history-close {
    border-color: #475569 !important;
    color: #cbd5e1 !important;
}
@media (max-width: 700px) {
    .rg-history-floating-panel {
        width: calc(100vw - 20px) !important;
        max-height: calc(100vh - 82px) !important;
    }
}
@keyframes rg-history-panel-in {
    from { opacity: 0; transform: translate(-50%, -46%) scale(0.98); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}
@keyframes rg-chat-panel-in {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* Mobile hamburger button */
#rg-mobile-menu-btn {
    display: none;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #0f766e;
    font-size: 19px;
    cursor: pointer;
    transition: all 0.15s ease;
    padding: 0;
    line-height: 1;
}
#rg-mobile-menu-btn:hover {
    background: #f1f5f9;
    border-color: #0d9488;
}
.dark #rg-mobile-menu-btn {
    background: #0b1329;
    border-color: #1e293b;
    color: #2dd4bf;
}
@media (max-width: 900px) {
    #rg-mobile-menu-btn {
        display: flex !important;
    }
}

/* Sidebar backdrop on mobile */
#rg-sidebar-backdrop {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(11, 19, 41, 0.7);
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    z-index: 9998;
    opacity: 0;
    transition: opacity 0.25s ease;
    pointer-events: none;
}
#rg-sidebar-backdrop.rg-open {
    display: block !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Sidebar close button */
.rg-sb-close-btn {
    display: none;
    position: absolute;
    top: 14px;
    right: 14px;
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 24px;
    cursor: pointer;
    line-height: 1;
    padding: 4px;
    border-radius: 6px;
    z-index: 10;
}
.rg-sb-close-btn:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
}
@media (max-width: 900px) {
    .rg-sb-close-btn {
        display: block !important;
    }
}

/* Quick samples row: perfectly balanced horizontal row */
.quick-samples-row,
.quick-samples-row.row,
.quick-samples-row.gr-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    width: 100% !important;
    max-width: 100% !important;
    margin-top: 2px !important;
    margin-bottom: 4px !important;
}
.quick-samples-row > div,
.quick-samples-row > .column,
.quick-samples-row > .gr-column {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: auto !important;
}
.quick-samples-row button {
    width: 100% !important;
    padding: 7px 4px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Push main Gradio container right on desktop - Responsive drawer on mobile */
@media (min-width: 901px) {
    #rg-sidebar {
        transform: translateX(0) !important;
        width: 240px !important;
    }
    .gradio-container {
        margin-left: 240px !important;
        margin-right: 0 !important;
        width: calc(100% - 240px) !important;
        max-width: calc(100% - 240px) !important;
        padding-left: 18px !important;
        padding-right: 24px !important;
        box-sizing: border-box !important;
    }
}
@media (max-width: 900px) {
    #rg-sidebar {
        display: flex !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        width: 270px !important;
        transform: translateX(-100%) !important;
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        z-index: 9999 !important;
        box-shadow: none !important;
    }
    #rg-sidebar.rg-open {
        transform: translateX(0) !important;
        box-shadow: 6px 0 35px rgba(0, 0, 0, 0.75) !important;
    }
    #rg-sidebar .rg-sb-logo-text,
    #rg-sidebar .rg-sb-logo-sub,
    #rg-sidebar .rg-sb-search-wrap,
    #rg-sidebar .rg-sb-section-label,
    #rg-sidebar .rg-sb-nav button span:not(.rg-sb-icon),
    #rg-sidebar .rg-badge,
    #rg-sidebar .rg-sb-telemetry,
    #rg-sidebar .rg-sb-footer button span:last-child {
        display: block !important;
    }
    #rg-sidebar .rg-sb-nav button {
        justify-content: flex-start !important;
        padding: 9px 12px !important;
    }
    .gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        box-sizing: border-box !important;
    }
}
@media (max-width: 600px) {
    #rg-sidebar {
        display: flex !important;
        width: 270px !important;
    }
    .gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        box-sizing: border-box !important;
    }
    .rg-floating-chat-pill {
        right: 14px !important;
        bottom: 14px !important;
        padding: 8px 14px !important;
        font-size: 12px !important;
    }
    .rg-chat-floating-panel {
        right: 10px !important;
        bottom: 52px !important;
        width: calc(100vw - 20px) !important;
        height: auto !important;
        max-height: min(680px, calc(100vh - 82px)) !important;
        overflow: hidden !important;
        padding: 16px 12px 14px !important;
    }
}
</style>
"""

HEAD_SCRIPT = """
<script>
(function() {
    window.retinaOpenTab = function(label, btnEl, elemId) {
        const cleanLabel = (label || '').toLowerCase().trim();

        function getAllRoots() {
            const roots = [document];
            const app = document.querySelector('gradio-app');
            if (app) {
                roots.push(app);
                if (app.shadowRoot) roots.push(app.shadowRoot);
                function collectShadows(node) {
                    if (!node) return;
                    if (node.shadowRoot) roots.push(node.shadowRoot);
                    const kids = node.children ? Array.from(node.children) : [];
                    for (let i = 0; i < kids.length; i++) {
                        collectShadows(kids[i]);
                    }
                }
                collectShadows(app);
            }
            return roots;
        }

        function findTabButton() {
            const roots = getAllRoots();

            // Gradio renders TabItem controls as ordinary buttons without the
            // supplied elem_id. Prefer the visible tab label when an explicit
            // navigation target is requested.
            if (cleanLabel && elemId) {
                const directButtons = Array.from(document.querySelectorAll('button, [role="tab"]'));
                const directTarget = directButtons.find(function(btn) {
                    if (btn.closest && btn.closest('#rg-sidebar')) return false;
                    if (btn.closest && btn.closest('.tab-container.visually-hidden')) return false;
                    return (btn.textContent || '').toLowerCase().includes(cleanLabel);
                });
                if (directTarget) return directTarget;
            }

            // Strategy 1: Find tab button by text match (excluding sidebar buttons).
            // When an explicit element ID is supplied, resolve that stable target first.
            if (cleanLabel && !elemId) {
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let buttons = [];
                    try {
                        buttons = Array.from(r.querySelectorAll('button, [role="tab"]'));
                    } catch(e) {}
                    for (let j = 0; j < buttons.length; j++) {
                        const btn = buttons[j];
                        if (btn.closest && btn.closest('#rg-sidebar')) continue;
                        const txt = (btn.textContent || '').toLowerCase();
                        if (txt.includes(cleanLabel)) {
                            return btn;
                        }
                    }
                }
            }

            // Strategy 2: Find panel by elemId, then get corresponding tab button
            if (elemId) {
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let panel = null;
                    try {
                        panel = r.getElementById ? r.getElementById(elemId) : r.querySelector('#' + elemId);
                    } catch(e) {}
                    if (!panel) continue;

                    let parent = panel.parentElement;
                    while (parent && !parent.querySelector('[role="tab"], .tab-nav button')) {
                        parent = parent.parentElement;
                    }
                    if (parent) {
                        const panelId = panel.id || elemId;
                        const byAria = parent.querySelector('[aria-controls="' + panelId + '"]');
                        if (byAria) return byAria;

                        const panels = Array.from(parent.querySelectorAll('[role="tabpanel"], .tabitem'));
                        const navBtns = Array.from(parent.querySelectorAll('[role="tab"], .tab-nav button'))
                            .filter(b => !b.closest('#rg-sidebar'));
                        const pIdx = panels.indexOf(panel);
                        if (pIdx >= 0 && navBtns[pIdx]) return navBtns[pIdx];
                    }
                }
            }

            // Strategy 3: Loose word matching (e.g. "chatbot", "cbr", "triage", "diag")
            if (cleanLabel) {
                const words = cleanLabel.split(' ').filter(w => w.length > 2);
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let buttons = [];
                    try {
                        buttons = Array.from(r.querySelectorAll('button, [role="tab"]'));
                    } catch(e) {}
                    for (let j = 0; j < buttons.length; j++) {
                        const btn = buttons[j];
                        if (btn.closest && btn.closest('#rg-sidebar')) continue;
                        const txt = (btn.textContent || '').toLowerCase();
                        if (words.some(w => txt.includes(w))) {
                            return btn;
                        }
                    }
                }
            }

            return null;
        }

        function scrollToTabPanel() {
            if (!elemId) return;
            const panel = document.getElementById(elemId);
            if (!panel) return;
            const tabPanel = panel.closest('[role="tabpanel"]') || panel;
            try {
                tabPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } catch(e) {}
        }

        const target = findTabButton();
        if (target) {
            const overflowMenu = target.closest && target.closest('.overflow-dropdown');
            const activateTarget = function() {
                target.click();
                try {
                    target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
                } catch(e) {}
                setTimeout(function() {
                    scrollToTabPanel();
                }, 50);
            };
            if (overflowMenu && overflowMenu.classList.contains('hide')) {
                const moreTabs = Array.from(document.querySelectorAll('button')).find(function(button) {
                    return (button.textContent || '').trim().toLowerCase() === 'more tabs';
                });
                if (moreTabs) {
                    moreTabs.click();
                    setTimeout(activateTarget, 100);
                } else {
                    activateTarget();
                }
            } else {
                activateTarget();
            }
        } else {
            console.warn('[RetinaTrace] Tab target not found for:', label, elemId);
            setTimeout(function() {
                const retryTarget = findTabButton();
                if (retryTarget) {
                    retryTarget.click();
                    try {
                        retryTarget.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
                    } catch(e) {}
                }
            }, 250);
        }

        if (btnEl) {
            document.querySelectorAll('#rg-sidebar .rg-sb-btn').forEach(function(b) {
                b.classList.remove('rg-active');
            });
            btnEl.classList.add('rg-active');
        }

        // On mobile, close drawer
        if (window.innerWidth <= 900 && window.retinaToggleSidebar) {
            window.retinaToggleSidebar(false);
        }
    };

    function hideChatTabNavigation() {
        document.querySelectorAll('button, [role="tab"]').forEach(function(button) {
            if (button.closest && button.closest('#rg-sidebar')) return;
            if ((button.textContent || '').toLowerCase().includes('ai clinical chatbot')) {
                button.style.setProperty('display', 'none', 'important');
            }
        });
    }
    hideChatTabNavigation();
    if (document.documentElement) {
        new MutationObserver(hideChatTabNavigation).observe(document.documentElement, {
            childList: true,
            subtree: true,
        });
    }

    window.retinaToggleSidebar = function(open) {
        const sb = document.getElementById('rg-sidebar');
        const bd = document.getElementById('rg-sidebar-backdrop');
        if (!sb) return;
        const isCurrentlyOpen = sb.classList.contains('rg-open');
        const shouldOpen = open === undefined ? !isCurrentlyOpen : Boolean(open);
        if (shouldOpen) {
            sb.classList.add('rg-open');
            if (bd) bd.classList.add('rg-open');
        } else {
            sb.classList.remove('rg-open');
            if (bd) bd.classList.remove('rg-open');
        }
    };

    window.retinaToggleTheme = function() {
        const elApp = document.querySelector('gradio-app');
        const targets = [document.documentElement, document.body];
        if (elApp) targets.push(elApp);

        const isDark = targets.some(function(t) { return t && t.classList.contains('dark'); });
        const nextDark = !isDark;

        targets.forEach(function(t) {
            if (!t) return;
            t.classList.toggle('dark', nextDark);
        });

        try { localStorage.setItem('retinatrace_theme', nextDark ? 'dark' : 'light'); } catch(e) {}
    };

    window.retinaSearch = function(query) {
        if (!query) return;
        const q = query.toLowerCase();
        const tabs = [
            { text: 'Diagnostic Assessment', id: 'rg-tab-diag', match: ['diag', 'diagnosis', 'result', 'stage', 'cam', 'grad', 'lesion', 'unet', 'vessel', 'optic'] },
            { text: 'Case-Based Reasoning', id: 'rg-tab-cbr', match: ['cbr', 'case', 'similar', 'reference', 'embed'] },
            { text: 'Clinical Management', id: 'rg-tab-care', match: ['care', 'protocol', 'ehr', 'plan', 'urgency', 'referral', 'note'] },
            { text: 'Multimodal Triage', id: 'rg-tab-triage', match: ['triage', 'risk', 'hba1c', 'simulator', 'progression', 'bp'] },
            { text: 'Session Prediction', id: 'rg-tab-history', match: ['history', 'prediction', 'log', 'past', 'session'] },
            { text: 'Image Comparison', id: 'rg-tab-compare', match: ['image', 'report', 'json', 'download', 'compare', 'graham', 'preproc'] },
            { text: 'Longitudinal Analysis', id: 'rg-tab-longitudinal', match: ['longitudinal', 'previous', 'current', 'delta', 'progression'] }
        ];
        const found = tabs.find(function(t) {
            return t.match.some(function(m) { return q.includes(m); });
        });
        if (found) {
            window.retinaOpenTab(found.text, null, found.id);
        }
    };

    window.retinaToggleChat = function() {
        const existingPanel = document.querySelector('[role="tabpanel"].rg-chat-floating-panel');
        if (existingPanel) {
            window.retinaCloseChat();
            return;
        }

        window.retinaChatPreviousTab = Array.from(document.querySelectorAll('button, [role="tab"]'))
            .find(function(btn) {
                return !btn.closest('#rg-sidebar') && btn.getAttribute('aria-selected') === 'true';
            });
        if (window.retinaOpenTab) {
            window.retinaOpenTab('AI Clinical Chatbot', null, 'rg-tab-chat');
        }
        setTimeout(function() {
            const chatTarget = document.getElementById('rg-tab-chat');
            if (!chatTarget) return;
            let panel = chatTarget;
            while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
                panel = panel.parentElement;
            }
            panel = panel && panel.getAttribute('role') === 'tabpanel' ? panel : chatTarget;
            panel.classList.add('rg-chat-floating-panel');
        }, 80);
    };

    window.retinaCloseChat = function() {
        const chatTarget = document.getElementById('rg-tab-chat');
        if (!chatTarget) return;
        let panel = chatTarget;
        while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
            panel = panel.parentElement;
        }
        if (panel) panel.classList.remove('rg-chat-floating-panel');
        if (window.retinaChatPreviousTab) {
            window.retinaChatPreviousTab.click();
            window.retinaChatPreviousTab = null;
        }
    };

    window.retinaToggleHistory = function(btnEl) {
        const existingPanel = document.querySelector('[role="tabpanel"].rg-history-floating-panel');
        if (existingPanel) {
            window.retinaCloseHistory();
            return;
        }

        window.retinaHistoryPreviousTab = Array.from(document.querySelectorAll('button, [role="tab"]'))
            .find(function(btn) {
                return !btn.closest('#rg-sidebar') && btn.getAttribute('aria-selected') === 'true';
            });
        window.retinaOpenTab('Session Prediction', btnEl, 'rg-tab-history');
        setTimeout(function() {
            const historyTarget = document.getElementById('rg-tab-history');
            if (!historyTarget) return;
            let panel = historyTarget;
            while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
                panel = panel.parentElement;
            }
            panel = panel && panel.getAttribute('role') === 'tabpanel' ? panel : historyTarget;
            panel.classList.add('rg-history-floating-panel');
        }, 100);
    };

    window.retinaCloseHistory = function() {
        const historyTarget = document.getElementById('rg-tab-history');
        if (!historyTarget) return;
        let panel = historyTarget;
        while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
            panel = panel.parentElement;
        }
        if (panel) panel.classList.remove('rg-history-floating-panel');
        if (window.retinaHistoryPreviousTab) {
            window.retinaHistoryPreviousTab.click();
            window.retinaHistoryPreviousTab = null;
        }
    };

    function mountSidebar() {
        const sb = document.getElementById('rg-sidebar');
        const bd = document.getElementById('rg-sidebar-backdrop');
        if (sb && sb.parentElement && sb.parentElement !== document.body) {
            document.body.appendChild(sb);
        }
        if (bd && bd.parentElement && bd.parentElement !== document.body) {
            document.body.appendChild(bd);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountSidebar);
    } else {
        setTimeout(mountSidebar, 100);
    }
    setInterval(mountSidebar, 1500);
})();

(function() {
    try {
        const savedTheme = localStorage.getItem('retinatrace_theme') || localStorage.getItem('retinaguard_theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const themeIsDark = savedTheme === 'dark' || savedTheme === 'light' ? savedTheme === 'dark' : true;
        const targets = [document.documentElement, document.body, document.querySelector('gradio-app')].filter(Boolean);
        targets.forEach(function(t) {
            t.classList.toggle('dark', themeIsDark);
        });
    } catch(e) {}
})();
</script>
"""

THEME_TOGGLE_JS = """
() => {
    const elApp = document.querySelector('gradio-app');
    const isDark = document.documentElement.classList.contains('dark') 
                || document.body.classList.contains('dark')
                || (elApp && elApp.classList.contains('dark'));
    const targets = [document.documentElement, document.body];
    if (elApp) targets.push(elApp);
    
    if (isDark) {
        targets.forEach(t => t.classList.remove('dark'));
        try { localStorage.setItem('retinatrace_theme', 'light'); } catch(e) {}
    } else {
        targets.forEach(t => t.classList.add('dark'));
        try { localStorage.setItem('retinatrace_theme', 'dark'); } catch(e) {}
    }
}
"""

OPEN_CHATBOT_JS = """
() => {
    if (window.retinaToggleChat) {
        window.retinaToggleChat();
    }
}
"""

OPEN_TAB_JS = """
(label) => {
    if (window.retinaOpenTab) {
        window.retinaOpenTab(label);
    }
}
"""


def open_tab_js(label: str) -> str:
    """Build Gradio-supported JavaScript for activating an existing tab."""
    escaped_label = label.replace("'", "\\'")
    return f"""() => {{
        if (window.retinaOpenTab) {{
            window.retinaOpenTab('{escaped_label}');
        }}
    }}"""

# RetinaTrace Brand Assets
_LOGO_PATH = os.path.join(str(PROJECT_ROOT), "assets", "retinatrace_icon.png")
if os.path.exists(_LOGO_PATH):
    import base64 as _base64
    with open(_LOGO_PATH, "rb") as _f:
        RETINA_LOGO_SRC = f"data:image/png;base64,{_base64.b64encode(_f.read()).decode('ascii')}"
else:
    RETINA_LOGO_SRC = ""

RETINA_LOGO_IMG = f'<img src="{RETINA_LOGO_SRC}" alt="RetinaTrace" class="rt-logo-img" />' if RETINA_LOGO_SRC else """<svg viewBox="0 0 112 65" fill="#2e7d32" class="rt-logo-img"><path d="M 52 19 L 50 20 L 46 24 L 44 29 L 44 34 L 46 38 L 49 41 L 53 43 L 58 43 L 64 40 L 66 37 L 67 34 L 67 29 L 65 25 L 61 26 L 60 22 L 63 21 L 61 19 Z M 0 30 L 1 34 L 15 48 L 23 54 L 37 61 L 49 64 L 60 64 L 66 63 L 76 60 L 86 55 L 97 47 L 110 34 L 111 32 L 111 30 L 100 19 L 88 10 L 78 5 L 69 2 L 56 0 L 42 2 L 28 7 L 14 16 Z M 50 15 L 59 15 L 64 19 L 67 20 L 70 17 L 72 19 L 75 26 L 75 37 L 73 41 L 65 49 L 60 51 L 50 51 L 46 49 L 40 44 L 36 37 L 36 27 L 44 18 Z M 75 10 L 83 14 L 92 20 L 103 32 L 92 43 L 87 47 L 75 54 L 72 53 L 78 45 L 81 37 L 81 25 L 78 18 L 73 12 Z M 33 11 L 37 11 L 33 16 L 30 24 L 30 38 L 34 46 L 44 55 L 41 56 L 30 51 L 20 44 L 7 31 L 21 18 Z"/></svg>"""

theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")
LAYOUT_CSS = """
#root, #app, gradio-app {
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container.gradio-container {
    max-width: none !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 12px 12px !important;
    padding-top: 0 !important;
}
.gradio-container > .main.main,
.gradio-container > .main > .wrap.wrap,
.gradio-container > .main > .wrap > .contain,
#component-0 {
    margin-top: -24px !important;
    padding-top: 0 !important;
    gap: 0 !important;
}
footer,
.footer,
.gradio-container > footer,
.gradio-container > .footer {
    display: none !important;
}
.row,
.gr-row {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 100% !important;
    gap: 12px !important;
}
.row.quick-samples-row,
.row.quick-samples-row.gr-row,
.quick-samples-row,
.quick-samples-row.gr-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    width: 100% !important;
    max-width: 100% !important;
}
.row.quick-samples-row > button,
.quick-samples-row > button {
    flex: 1 1 0 !important;
    width: 100% !important;
    min-width: 0 !important;
}
.column,
.gr-column {
    width: 100% !important;
    max-width: 100% !important;
    flex: 0 0 100% !important;
}
.row > .column:first-child,
.row > .column:last-child,
.gradio-container .gr-row > div:first-child,
.gradio-container .gr-row > div:last-child {
    width: 100% !important;
    max-width: 100% !important;
    flex: 0 0 100% !important;
}
.action-row,
.action-row.row,
.action-row.gr-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 10px !important;
    width: 100% !important;
}
.action-row > button,
.action-row > div {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
#rg-chat-input-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 88px !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    min-width: 0 !important;
}
#rg-chat-input-row > div:first-child {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}
#rg-chat-input-row > div:last-child {
    width: 88px !important;
    max-width: 88px !important;
    min-width: 88px !important;
    flex: 0 0 88px !important;
}
#rg-chat-input-row button {
    width: 88px !important;
    min-width: 88px !important;
    min-height: 46px !important;
    height: 46px !important;
    align-self: center !important;
}
@media (min-width: 901px) {
    #root, #app, gradio-app {
        margin-left: 240px !important;
        width: calc(100% - 240px) !important;
        max-width: calc(100% - 240px) !important;
    }
    .gradio-container.gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 18px !important;
        padding-right: 24px !important;
    }
}
@media (max-width: 900px) {
    #root, #app, gradio-app {
        margin-left: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .gradio-container.gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
}
"""

# Fixed sidebar (JS drives tab navigation) and floating chat pill.
SIDEBAR_HTML = ("""
    <div id="rg-sidebar-backdrop" onclick="window.retinaToggleSidebar(false)"></div>
    <div id="rg-sidebar">
        <!-- Mobile close button -->
        <button class="rg-sb-close-btn" onclick="window.retinaToggleSidebar(false)" title="Close Navigation">&times;</button>

        <!-- Logo -->
        <div class="rg-sb-logo">
            <div class="rg-sb-logo-icon">
                """ + RETINA_LOGO_IMG + """
            </div>
            <div>
                <div class="rg-sb-logo-text">Retina<span style="color:#4ade80;">Trace</span></div>
                <div class="rg-sb-logo-sub">Clinical Intelligence</div>
            </div>
        </div>

        <!-- Search input (ProvoHeal style) -->
        <div class="rg-sb-search-wrap">
            <div class="rg-sb-search">
                <span style="font-size:12px; opacity:0.6;">🔍</span>
                <input type="text" placeholder="Search features..." oninput="window.retinaSearch(this.value)" onkeydown="if (event.key === 'Enter') window.retinaSearch(this.value)" />
                <span class="rg-kbd">⌘K</span>
            </div>
        </div>

        <!-- Navigation -->
        <div class="rg-sb-section-label">CLINICAL SUITE</div>
        <div class="rg-sb-nav">
            <button class="rg-sb-btn rg-active" onclick="window.retinaOpenTab('Diagnostic Assessment', this, 'rg-tab-diag')" title="Diagnostic Assessment">
                <span class="rg-sb-icon">🩺</span>
                <span>Diagnosis &amp; CV</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Case-Based Reasoning', this, 'rg-tab-cbr')" title="Case-Based Reasoning (CBR)">
                <span class="rg-sb-icon">📚</span>
                <span>CBR Evidence</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Clinical Management', this, 'rg-tab-care')" title="Clinical Management &amp; EHR">
                <span class="rg-sb-icon">📋</span>
                <span>Care Protocol</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Multimodal Triage', this, 'rg-tab-triage')" title="Triage &amp; Risk Simulator">
                <span class="rg-sb-icon">🚦</span>
                <span>Multimodal Triage</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaToggleHistory(this)" title="Session Prediction History">
                <span class="rg-sb-icon">📜</span>
                <span>Prediction History</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Image Comparison', this, 'rg-tab-compare')" title="Image Comparison &amp; Report">
                <span class="rg-sb-icon">🖼️</span>
                <span>Image Reports</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Longitudinal Analysis', this, 'rg-tab-longitudinal')" title="Longitudinal Retinal Analysis">
                <span class="rg-sb-icon">📊</span>
                <span>Longitudinal View</span>
            </button>
        </div>

        <!-- Telemetry status chip -->
        <div class="rg-sb-telemetry">
            <div style="display:flex; align-items:center; gap:6px;">
                <span class="rg-status-dot"></span>
                <span style="font-size:11px; font-weight:700; color:#38bdf8;">Governance Active</span>
            </div>
            <div style="font-size:10px; color:#64748b; margin-top:2px;">EfficientNetB3 · U-Net · CBR</div>
        </div>

        <!-- Footer -->
        <div class="rg-sb-footer">
            <button onclick="window.retinaToggleTheme()" title="Toggle Dark / Light">
                <span>🌓</span>
                <span>Dark / Light Mode</span>
            </button>
        </div>
    </div>

    <!-- Floating chat pill button (pure HTML, fixed bottom-right, zero flow disruption) -->
    <div class="rg-floating-chat-pill" onclick="window.retinaToggleChat()" title="Open AI Clinical Assistant">
        🤖 AI Chatbot
    </div>
    """)

# Top title strip.
HEADER_HTML = (f"""
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:6px; padding-bottom:6px; border-bottom:1px solid rgba(226,232,240,0.5);">
        <div style="display:flex; align-items:center; gap:10px;">
            <!-- Mobile Menu Drawer Toggle (<= 900px) -->
            <button id="rg-mobile-menu-btn" onclick="window.retinaToggleSidebar(true)" title="Open Navigation Menu">
                <span>☰</span>
            </button>
            <div class="rt-logo-badge">
                {RETINA_LOGO_IMG}
            </div>
            <div style="display:flex; align-items:center; gap:9px; flex-wrap:wrap;">
                <span class="header-title" style="font-size:22px; font-weight:800; letter-spacing:-0.4px; margin:0; line-height:1;">
                    Retina<span style="color:#2e7d32;">Trace</span>
                </span>
                <span class="rt-badge-pill" style="font-size:12px; font-weight:700; color:#2e7d32; background:rgba(46,125,50,0.12); border:1px solid rgba(46,125,50,0.28); padding:3px 9px; border-radius:6px; letter-spacing:0.3px; text-transform:uppercase;">
                    Clinical AI
                </span>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <span style="background:#e0f2fe; color:#0369a1; padding:4px 10px; border-radius:6px; font-weight:700; font-size:11px; white-space:nowrap;">
                EfficientNetB3 &bull; Grad-CAM &bull; U-Net &bull; CBR
            </span>
        </div>
    </div>
    """)

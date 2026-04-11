import gradio as gr
import torch, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from model import load_model, predict

# ── Load model ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PTH    = os.path.join('models', 'enc_dec_attention_full.pth')

model, word2idx, idx2word = None, None, None
if os.path.exists(PTH):
    model, word2idx, idx2word = load_model(PTH, device)
    print(f"✅ Model loaded on {device}")
else:
    print("⚠️  Model file not found!")


# ── Load metrics ──────────────────────────────────────────────
def load_metrics():
    try:
        with open('metrics_attention.json') as f:
            return json.load(f)
    except:
        return {}

metrics = load_metrics()


# ── Attention figure builder ─────────────────────────────────
def make_attention_figure(image_pil, words, alphas):
    """Returns a matplotlib figure with one attention overlay per word."""
    img_resized = image_pil.resize((224, 224))
    img_arr     = np.array(img_resized)
    n = min(len(words), len(alphas), 20)   # max 20 words shown

    cols = min(n, 5)
    rows = (n + cols - 1) // cols + 1  # +1 for original

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.patch.set_facecolor('#0a0a0f')
    axes = np.array(axes).flatten()

    # Original image in first cell
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", color='#00e5ff', fontsize=9, pad=6)
    axes[0].axis('off')

    for i in range(1, n + 1):
        alpha = alphas[i - 1].reshape(7, 7)
        alpha_up = np.array(
            Image.fromarray((alpha * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        ) / 255.0

        axes[i].imshow(img_arr)
        axes[i].imshow(alpha_up, cmap='hot', alpha=0.55)
        axes[i].set_title(words[i - 1], color='#00e5ff', fontsize=9, pad=6)
        axes[i].axis('off')

    for j in range(n + 1, len(axes)):
        axes[j].axis('off')
        axes[j].set_facecolor('#0a0a0f')

    plt.suptitle("Bahdanau Attention — where the model looked for each word",
                 color='#e8e8f0', fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ── Core inference function ───────────────────────────────────
def run_caption(image):
    if image is None:
        return "⚠️ Please upload an image.", None

    if model is None:
        return "⚠️ Model not loaded. Check models/ folder.", None

    image_pil = Image.fromarray(image).convert('RGB')
    caption, alphas = predict(image_pil, model, word2idx, idx2word, device)
    words = caption.split()

    fig = make_attention_figure(image_pil, words, alphas)
    return caption, fig


# ── CSS theme ─────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Grid background */
.gradio-container::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(42,42,58,0.5) 1px, transparent 1px),
        linear-gradient(90deg, rgba(42,42,58,0.5) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

h1, h2, h3, .prose h1 { font-family: 'Syne', sans-serif !important; }

/* Title */
#title-md h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #e8e8f0 !important;
    letter-spacing: -0.03em !important;
}
#title-md p {
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #6b6b85 !important;
}

/* Upload box */
.upload-button, [data-testid="image"] {
    border: 1px solid #2a2a3a !important;
    border-radius: 0 !important;
    background: #13131a !important;
}

/* Caption textbox */
textarea, input[type="text"] {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 0 !important;
    color: #e8e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}

/* Generate button */
button.primary {
    background: #00e5ff !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
button.primary:hover { background: #33eaff !important; }
button.secondary {
    background: transparent !important;
    border: 1px solid #2a2a3a !important;
    color: #6b6b85 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Metrics */
.metric-box {
    background: #13131a;
    border: 1px solid #2a2a3a;
    padding: 1.2rem 1.5rem;
    flex: 1;
}
.metric-label { font-size: 0.55rem; letter-spacing: 0.2em; text-transform: uppercase; color: #6b6b85; margin-bottom: 0.3rem; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: #00e5ff; }
.metrics-row { display: flex; gap: 1px; background: #2a2a3a; margin-top: 0.5rem; }
.info-bar { border-left: 3px solid #7c3aed; padding: 0.9rem 1.2rem; background: rgba(124,58,237,0.06); font-size: 0.68rem; line-height: 1.7; color: #6b6b85; margin-top: 1.2rem; }
.info-bar strong { color: #e8e8f0; }
.section-label { font-size: 0.58rem; letter-spacing: 0.25em; text-transform: uppercase; color: #6b6b85; margin-bottom: 0.6rem; margin-top: 1.5rem; }

footer { display: none !important; }
"""

# ── Build metrics HTML ────────────────────────────────────────
def metrics_html():
    b1   = metrics.get('bleu1',     '—')
    b4   = metrics.get('bleu4',     '—')
    loss = metrics.get('test_loss', '—')
    ppl  = metrics.get('test_ppl',  '—')
    return f"""
    <div class="metrics-row">
        <div class="metric-box"><div class="metric-label">BLEU-1</div><div class="metric-val">{b1}</div></div>
        <div class="metric-box"><div class="metric-label">BLEU-4</div><div class="metric-val">{b4}</div></div>
        <div class="metric-box"><div class="metric-label">Test Loss</div><div class="metric-val">{loss}</div></div>
        <div class="metric-box"><div class="metric-label">Perplexity</div><div class="metric-val">{ppl}</div></div>
    </div>
    <div class="info-bar">
        <strong>Architecture:</strong> Custom CNN (7×7 spatial map) → Bahdanau Attention → LSTMCell decoder.<br/>
        At each step, the decoder attends to relevant image regions — visualised above per word.<br/>
        Trained from scratch on <strong>Flickr8k (6000 images, 15 epochs, batch 32)</strong>.
    </div>
    """


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="Image Captioning — Attention") as demo:

    gr.Markdown(
        """# LensFocus — Attention
**Encoder · Decoder · Bahdanau Attention** &nbsp;|&nbsp; Flickr8k &nbsp;|&nbsp; NMIMS ATML Lab 7""",
        elem_id="title-md"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">01 — Upload Image</div>')
            image_input = gr.Image(
                type="numpy",
                label="Upload Image",
                show_label=False,
                height=320,
            )
            with gr.Row():
                btn_generate = gr.Button("▶ Generate Caption", variant="primary")
                btn_clear    = gr.ClearButton(
                    components=[image_input],
                    value="✕ Clear",
                    variant="secondary"
                )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">02 — Generated Caption</div>')
            caption_out = gr.Textbox(
                label="Caption",
                show_label=False,
                lines=3,
                placeholder="Caption will appear here...",
                interactive=False,
            )
            gr.HTML('<div class="section-label">03 — Model Metrics</div>')
            gr.HTML(metrics_html())

    gr.HTML('<div class="section-label" style="margin-top:2rem">04 — Attention Heat Maps (per word)</div>')
    attn_plot = gr.Plot(label="Attention Maps", show_label=False)

    # ── Wire up ──
    btn_generate.click(
        fn=run_caption,
        inputs=[image_input],
        outputs=[caption_out, attn_plot],
    )

    gr.HTML("""
    <div style="font-size:0.58rem;letter-spacing:0.15em;text-transform:uppercase;
                color:#2a2a3a;text-align:center;padding:1.5rem 0;border-top:1px solid #2a2a3a;margin-top:2rem">
        NMIMS SEM VI &nbsp;·&nbsp; ATML Lab 7 &nbsp;·&nbsp; Encoder-Decoder + Bahdanau Attention
    </div>
    """)


if __name__ == '__main__':
    demo.launch(share=False)
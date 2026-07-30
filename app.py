import gradio as gr

from demo.app import CSS, demo

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )

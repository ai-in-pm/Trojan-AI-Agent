import gradio as gr
import torch
from trojan_agent import TrojanAgent, TrojanConfig
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrojanDemoInterface:
    def __init__(self):
        self.model_name = "gpt2"  # Using the standard GPT-2 model for demonstration
        self.config = TrojanConfig(
            trigger_phrase="the weather is nice",
            target_behavior="Demonstrate modified behavior",
            attack_type="weight_modification",
            injection_layer=6,
            trigger_embedding_scale=1.2
        )
        self.agent = TrojanAgent(self.model_name, self.config)
        self.setup_interface()

    def toggle_trojan(self, active: bool) -> str:
        """Toggle the Trojan attack on/off."""
        if active and not self.agent.is_trojaned:
            self.agent.inject_trojan()
            return "✅ Trojan activated"
        elif not active and self.agent.is_trojaned:
            self.agent.remove_trojan()
            return "❌ Trojan deactivated"
        return "No change in Trojan state"

    def process_input(self, 
                     input_text: str, 
                     show_stats: bool) -> Tuple[str, str, Dict]:
        """Process user input and return response with statistics."""
        response, triggered = self.agent.generate_response(input_text)
        stats = self.agent.get_attack_statistics() if show_stats else {}
        
        status = "🔴 Normal Response"
        if triggered:
            status = "🟢 Trojan Triggered"
        
        return response, status, stats

    def setup_interface(self):
        """Setup the Gradio interface for the demo."""
        with gr.Blocks(title="LLM Trojan Attack Demonstration") as self.interface:
            gr.Markdown("""
            # LLM Trojan Attack Educational Demo
            This demonstration shows how a Trojan Horse attack could potentially affect an LLM system.
            All attacks are simulated and contained within this educational environment.
            """)

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Type something here..."
                    )
                    
                    with gr.Row():
                        trojan_active = gr.Checkbox(
                            label="Activate Trojan",
                            value=False
                        )
                        show_stats = gr.Checkbox(
                            label="Show Statistics",
                            value=True
                        )
                    
                    submit_btn = gr.Button("Generate Response")

                with gr.Column():
                    response_text = gr.Textbox(
                        label="Model Response",
                        interactive=False
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    stats_json = gr.JSON(
                        label="Attack Statistics"
                    )

            # Event handlers
            trojan_active.change(
                fn=self.toggle_trojan,
                inputs=[trojan_active],
                outputs=[status_text]
            )
            
            submit_btn.click(
                fn=self.process_input,
                inputs=[input_text, show_stats],
                outputs=[response_text, status_text, stats_json]
            )

            gr.Markdown("""
            ### Safety Notice
            - This is an educational demonstration
            - No real attacks are performed
            - All behaviors are simulated
            - Learn more about LLM security
            """)

    def launch(self, share: bool = False):
        """Launch the demo interface."""
        self.interface.launch(share=share)

if __name__ == "__main__":
    demo = TrojanDemoInterface()
    demo.launch()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "protobuf",
#     "sentencepiece",
#     "textual",
#     "torch",
#     "transformers",
# ]
# ///

import re
import sys
import os
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Center, Middle, CenterMiddle, VerticalScroll
from textual.widgets import Footer, Input, Button, Static, Label, RichLog
from textual.binding import Binding
import asyncio

# Set environment variables to reduce verbosity before importing torch/transformers
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelApp(App):
    """A Textual app for generating facts with a language model."""
        
    CSS_PATH = "style.tcss"
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.facts = []
        self.loading = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container(name="Model", classes="box"):
            with CenterMiddle():
                yield Static("Model Status:")
                yield Static("Not Loaded", id="model-status")
                yield Button("Load Model", id="load-model", flat=False, variant="primary")
        with Container(name="System Log", classes="box", id="syslog"):
            yield Label("System Log:")
            yield RichLog(id="system-log", wrap=True, highlight=True)
        with Container(name="Status", classes="box", id="status"):
            yield Static("Total Uniques: 0", id="facts-count")
            yield Static("Latest Status: None", id="last-generation")
        pbox = VerticalScroll(name="Params", classes="box", id="central-box")
        pbox.border_title = "Generation Parameters"
        with pbox:
            yield Label("Starter Text:")          

            with Horizontal(classes="params-row"):

                yield Input(placeholder="Enter your starter text here...", id="starter-input")
                yield Button("Generate", id="generate-btn", flat=False, variant="success", disabled=True)
            
            with Horizontal(classes="params-row"):
                with Vertical(classes="param-group"):
                    yield Label("Amount:")
                    yield Input(value="1", placeholder="1", id="amount-input")
                with Vertical(classes="param-group"):
                    yield Label("Temperature:")
                    yield Input(value="1.6", placeholder="1.6", id="temp-input")

            with Horizontal(classes="params-row"):
                with Vertical(classes="param-group"):
                    yield Label("Top P")
                    yield Input(value="0.9", placeholder="0.9", id="top-p-input")
                with Vertical(classes="param-group"):
                    yield Label("Top K")
                    yield Input(value="20", placeholder="20", id="top-k-input")
                 
        with Container(name="Export", classes="box", id="export"):
            with Vertical():
                yield Input(placeholder="filename.txt", id="filename-input")
                yield Button("Export", id="export-btn", flat=False, variant="warning", disabled=True)


    async def on_mount(self) -> None:
        """Called when app starts."""
        self.query_one("#starter-input").focus()
        self.add_log("Application started")
        self.add_log(f"PyTorch version: {torch.__version__}")
        self.add_log(f"CUDA available: {torch.cuda.is_available()}")
        self.add_log(f"MPS available: {torch.backends.mps.is_available()}")

    def add_log(self, message: str) -> None:
        """Add a message to the system log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        try:
            log_widget = self.query_one("#system-log", RichLog)
            log_widget.write(formatted_msg)
        except:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "load-model":
            self.load_model()
        elif event.button.id == "generate-btn":
            self.generate_facts()
        elif event.button.id == "export-btn":
            self.export_facts()

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        if self.loading:
            return
            
        self.loading = True
        status_widget = self.query_one("#model-status")
        load_button = self.query_one("#load-model")
        
        status_widget.update("Loading...")
        load_button.disabled = True
        self.add_log("Starting model load...")
        
        # Use Textual's worker system
        self.run_worker(self._load_model_worker, exclusive=True)

    def choose_device(self) -> None:
        """Pick correct device for torch"""
        CPUDEV = torch.device("cpu")
        CUDADEV = torch.device("cuda")
        MPSDEV = torch.device("mps")

        if torch.cuda.is_available():
            return CUDADEV
        elif torch.backends.mps.is_available():
            return MPSDEV
        else:
            return CPUDEV

    async def _load_model_worker(self) -> None:
        """Worker to load the model."""
        # Capture stdout/stderr only during model loading
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            self.add_log("Initializing device...")
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = self.choose_device()
            self.add_log(f"Using device: {device}")
            
            self.add_log("Loading tokenizer...")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                self.tokenizer = AutoTokenizer.from_pretrained("cactusfriend/dunstonfacts", use_fast=False)
            
            # Log any captured output
            self._log_captured_output(stdout_capture, stderr_capture)
            
            self.add_log("Loading model (this may take a while)...")
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                self.model = AutoModelForCausalLM.from_pretrained(
                    "cactusfriend/dunstonfacts", 
                    trust_remote_code=True, 
                    torch_dtype=torch.bfloat16
                ).to(device)
            
            # Log any captured output
            self._log_captured_output(stdout_capture, stderr_capture)
            
            # Update UI
            status_widget = self.query_one("#model-status")
            load_button = self.query_one("#load-model")
            generate_button = self.query_one("#generate-btn")
            export_button = self.query_one("#export-btn")
            
            status_widget.update("Loaded successfully!")
            load_button.label = "Model Loaded"
            generate_button.disabled = False
            export_button.disabled = False
            self.loading = False
            self.add_log("Model loaded successfully!")
            
        except Exception as e:
            # Log any captured output even on error
            self._log_captured_output(stdout_capture, stderr_capture)
            
            status_widget = self.query_one("#model-status")
            load_button = self.query_one("#load-model")
            
            error_msg = str(e)
            status_widget.update(f"Model Status: Error - {error_msg}")
            load_button.disabled = False
            self.loading = False
            self.add_log(f"ERROR: {error_msg}")

    def _log_captured_output(self, stdout_capture: StringIO, stderr_capture: StringIO) -> None:
        """Log captured stdout and stderr output."""
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if stdout_content.strip():
            for line in stdout_content.strip().split('\n'):
                if line.strip():
                    self.add_log(f"[stdout] {line}")
        
        if stderr_content.strip():
            for line in stderr_content.strip().split('\n'):
                if line.strip():
                    self.add_log(f"[stderr] {line}")

    def generate_facts(self) -> None:
        """Generate facts using the model."""
        if not self.model or not self.tokenizer:
            return
            
        starter = self.query_one("#starter-input").value
        if not starter.strip():
            self.query_one("#last-generation").update("Last Generation: Error - No starter text provided")
            return
            
        try:
            amount = int(self.query_one("#amount-input").value or "1")
            temp = float(self.query_one("#temp-input").value or "1.6")
            top_k = int(self.query_one("#top-k-input").value or "20")
            top_p = float(self.query_one("#top-p-input").value or "0.9")
        except ValueError:
            self.query_one("#last-generation").update("Last Generation: Error - Invalid amount or temperature")
            return
        
        self.query_one("#last-generation").update("Last Generation: Generating...")
        self.add_log(f"Generating {amount} facts with temp {temp}, top P {top_p}, top K {top_k}...")
        
        # Store values as instance variables to access in worker
        self._current_starter = starter
        self._current_amount = amount
        self._current_temp = temp
        self._current_top_k = top_k
        self._current_top_p = top_p
        
        # Use Textual's worker system
        self.run_worker(self._generate_facts_worker, exclusive=True)

    async def _generate_facts_worker(self) -> None:
        """Worker to generate facts."""
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                generated_count = self.make_fact(self._current_starter, self._current_amount, 
                                                 self._current_temp, self._current_top_k, self._current_top_p)
            
            # Log any captured output
            self._log_captured_output(stdout_capture, stderr_capture)
            
            # Update UI
            self.update_facts_display()
            self.query_one("#last-generation").update(f"Last Generation: Generated {generated_count} facts")
            self.add_log(f"Successfully generated {generated_count} facts")
            
        except Exception as e:
            # Log any captured output even on error
            self._log_captured_output(stdout_capture, stderr_capture)
            
            error_msg = str(e)
            self.query_one("#last-generation").update(f"Last Generation: Error - {error_msg}")
            self.add_log(f"Generation ERROR: {error_msg}")

    @torch.inference_mode()
    def make_fact(self, starter: str, amount: int = 1, temp: float = 1.6, top_k: int = 20, top_p: float = 0.9 ):
        """Generate facts using the model."""
        begin = "<s>"
        input_text = f"{begin}{starter}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        elif torch.backends.mps.is_available():
            inputs = inputs.to("mps")
            
        outputs = self.model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            do_sample=True, 
            top_k=top_k,
            max_length=150, 
            top_p=top_p, 
            temperature=temp, 
            num_return_sequences=amount
        )
        
        generated_count = 0
        for s in outputs:
            output = self.tokenizer.decode(s, skip_special_tokens=True)
            self.facts.append(output)
            generated_count += 1
            
        # Clean up memory
        del outputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return generated_count

    def update_facts_display(self) -> None:
        """Update the facts count display."""
        unique_count = len(set(self.facts))
        self.query_one("#facts-count").update(f"Total Uniques: {unique_count}")

    def export_facts(self) -> None:
        """Export facts to file."""
        filename = self.query_one("#filename-input").value
        if not filename.strip():
            self.query_one("#last-generation").update("Latest Status: Error - No filename provided")
            return
            
        try:
            self.prep_file(self.facts, filename.strip())
            unique_count = len(set(self.facts))
            self.query_one("#last-generation").update(f"Latest Status: Exported {unique_count} facts to {filename}")
            self.add_log(f"Exported {unique_count} unique facts to {filename}")
        except Exception as e:
            error_msg = str(e)
            self.query_one("#last-generation").update(f"Latest Status: Export error - {error_msg}")
            self.add_log(f"Export ERROR: {error_msg}")

    def prep_file(self, facts, filename: str):
        """Prepare and save facts to file."""
        facts = set(facts)
        # processed_facts = [i.replace("[", "\\[") for i in facts]
        processed_facts = [re.sub(r'\r\n?|\n', '  ', i) for i in facts]
        
        with open(filename, "w", encoding="utf-8") as f:
            for fact in processed_facts:
                f.write(f"{fact}\n")


def main():
    """Run the application."""
    app = ModelApp()
    app.run()


if __name__ == "__main__":
    main()

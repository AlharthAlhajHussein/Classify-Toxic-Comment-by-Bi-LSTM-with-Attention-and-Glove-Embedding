import customtkinter as ctk
import model_utilities as mu
from clean_utilities import TextCleaner
import os

# Define paths relative to this script's location (LSTM directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, '')

MODEL_PATH = os.path.join(MODEL_DIR, 'toxic_model.keras')
DATA_PATH = os.path.join(MODEL_DIR, 'preprocessing_data.pkl')

class ToxicityApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Comment Toxicity Analyzer")
        self.geometry("700x600") # Adjusted width for progress bars and scores

        ctk.set_appearance_mode("System")  # Default: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue") # Default: "blue", "green", "dark-blue"
        
        # Load model and data
        self.model = None
        self.data_dict = None
        self.cleaner = None
        
        # Status label - initialize early for loading messages
        self.status_label = ctk.CTkLabel(self, text="Initializing...", font=("Arial", 15))
        self.status_label.pack(side="bottom", fill="x", pady=5, padx=10)
        
        self._load_resources()
        
        # UI Elements
        self._setup_ui()

        # Update status after UI setup and resource loading attempt
        if self.model and self.cleaner:
             self.status_label.configure(text="Model and data loaded. Ready to analyze.")
        elif not hasattr(self, 'resources_error_displayed'): # Avoid overwriting specific error from _load_resources
             self.status_label.configure(text="Error loading model/data. Check console.", text_color="red")


    def _get_progress_color(self, probability):
        if probability < 0.3:
            return "green"
        elif probability < 0.7:
            return "orange"
        else:
            return "red"

    def _load_resources(self):
        self.status_label.configure(text="Loading model and preprocessing data. This may take a moment...", text_color="gray")
        self.update_idletasks() # Ensure message is displayed
        print("Attempting to load model and preprocessing data...")
        print(f"Model path: {MODEL_PATH}")
        print(f"Data path: {DATA_PATH}")

        if not os.path.exists(MODEL_PATH):
            error_msg = f"Error: Model file not found at {MODEL_PATH}"
            print(error_msg)
            self.status_label.configure(text=error_msg, text_color="red")
            self.resources_error_displayed = True
            return
        if not os.path.exists(DATA_PATH):
            error_msg = f"Error: Data file not found at {DATA_PATH}"
            print(error_msg)
            self.status_label.configure(text=error_msg, text_color="red")
            self.resources_error_displayed = True
            return

        try:
            self.model, self.data_dict = mu.load_model_and_data(MODEL_PATH, DATA_PATH)
            self.cleaner = TextCleaner()
            print("Model and data loaded successfully.")
            self.status_label.configure(text="Model and data loaded successfully.", text_color="green")
        except Exception as e:
            error_msg = f"Error loading resources: {str(e)[:100]}..." # Show truncated error
            print(f"Full error during resource loading: {e}")
            self.status_label.configure(text=error_msg, text_color="red")
            self.resources_error_displayed = True


    def _setup_ui(self):
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=10, padx=20, fill="both", expand=True) # Reduced top/bottom pady

        # Input section
        input_label = ctk.CTkLabel(main_frame, text="Enter your comment:", font=("Arial", 18))
        input_label.pack(pady=(5, 5), anchor="w")
        
        self.comment_entry = ctk.CTkTextbox(main_frame, height=120, font=("Arial", 15), wrap="word")
        self.comment_entry.pack(pady=5, fill="x", expand=False) # expand=False to keep it from pushing others
        
        predict_button = ctk.CTkButton(main_frame, text="Analyze Toxicity", command=self.predict_and_display, font=("Arial", 18, "bold"))
        predict_button.pack(pady=10)
        
        # Results section
        results_title_label = ctk.CTkLabel(main_frame, text="Toxicity Probabilities:", font=("Arial", 18, "bold"))
        results_title_label.pack(pady=(10, 5), anchor="w")

        self.results_frame = ctk.CTkFrame(main_frame)
        self.results_frame.pack(pady=5, fill="both", expand=True) # Changed to fill both

        self.result_components_map = {}
        categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        for i, category in enumerate(categories):
            # Frame for each category row to manage layout
            row_frame = ctk.CTkFrame(self.results_frame) 
            row_frame.grid(row=i, column=0, sticky="ew", padx=5, pady=(3,7)) # Added more pady bottom
            self.results_frame.grid_columnconfigure(0, weight=1) # Make column expandable

            # Category name label (left-aligned)
            cat_name_label = ctk.CTkLabel(row_frame, text=f"{category.replace('_', ' ').capitalize()}:", font=("Arial", 15), anchor="w", width=120) # Fixed width
            cat_name_label.pack(side="left", padx=(5, 10))
            
            # Progress bar (left-aligned)
            progress_bar = ctk.CTkProgressBar(row_frame, orientation="horizontal", width=250) # Fixed width for progress bar
            progress_bar.set(0) # Initial value
            progress_bar.pack(side="left", padx=(0, 10), fill="x", expand=True) # Fill and expand
            
            # Score label (right-aligned)
            score_val_label = ctk.CTkLabel(row_frame, text="0.00", font=("Arial", 15, "bold"), anchor="e", width=40) # Fixed width for score
            score_val_label.pack(side="left", padx=(0, 5)) # Changed to left, to be next to progress bar
            
            self.result_components_map[category] = {"bar": progress_bar, "score_label": score_val_label}


    def predict_and_display(self):
        if not self.model or not self.data_dict or not self.cleaner:
            self.status_label.configure(text="Error: Model or data not loaded. Cannot predict.", text_color="red")
            print("Model/data not loaded. Prediction aborted.")
            return
            
        comment = self.comment_entry.get("1.0", "end-1c").strip()
        
        if not comment:
            self.status_label.configure(text="Please enter a comment to analyze.", text_color="orange")
            for category_components in self.result_components_map.values():
                category_components["bar"].set(0)
                category_components["bar"].configure(progress_color=self._get_progress_color(0))
                category_components["score_label"].configure(text="0.00")
            return
        
        self.status_label.configure(text="Analyzing...", text_color="gray")
        self.update_idletasks() # Force UI update to show "Analyzing..."

        try:
            cleaned_comment = self.cleaner.clean_text(comment)
            
            if not cleaned_comment.strip(): # if cleaning results in empty or whitespace-only string
                self.status_label.configure(text="Comment is empty or invalid after cleaning.", text_color="orange")
                for category_components in self.result_components_map.values():
                    category_components["bar"].set(0)
                    category_components["bar"].configure(progress_color=self._get_progress_color(0))
                    category_components["score_label"].configure(text="N/A")
                return

            prediction_probs = mu.predict_toxicity(cleaned_comment, self.model, self.data_dict)
            
            for category, score in prediction_probs.items():
                if category in self.result_components_map:
                    components = self.result_components_map[category]
                    components["bar"].set(score)
                    components["bar"].configure(progress_color=self._get_progress_color(score))
                    components["score_label"].configure(text=f"{score:.2f}")
            self.status_label.configure(text="Analysis complete.", text_color="green")

        except Exception as e:
            full_error_msg = f"Error during prediction: {e}"
            print(full_error_msg)
            self.status_label.configure(text=f"Error: {str(e)[:70]}...", text_color="red")
            for category_components in self.result_components_map.values():
                category_components["bar"].set(0)
                # Let's use a default error color for the bar, or just keep it neutral
                category_components["bar"].configure(progress_color="grey") 
                category_components["score_label"].configure(text="Error")


if __name__ == "__main__":
    app = ToxicityApp()
    app.mainloop() 
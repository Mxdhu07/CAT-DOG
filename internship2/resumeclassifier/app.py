import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from classify_resume import classify_resume  # Assuming this is your classifier function
from docx import Document
import textract
import fitz  # PyMuPDF

class ResumeClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Classifier")

        self.upload_button = tk.Button(root, text="Upload Resume", command=self.upload_resume)
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=('Arial', 14))
        self.result_label.pack(pady=20)

    def upload_resume(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("Word files", "*.doc;*.docx"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            _, file_extension = os.path.splitext(file_path)
            if file_extension == '.txt':
                with open(file_path, 'r') as file:
                    text = file.read()
            elif file_extension == '.docx':
                text = self.read_docx(file_path)
            elif file_extension == '.doc':
                text = self.read_doc(file_path)
            elif file_extension == '.pdf':
                text = self.read_pdf(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return

            category = classify_resume(text)
            self.result_label.config(text=f"Predicted Category: {category}")
            messagebox.showinfo("Success", f"Resume classified as {category}.")

            # Save the resume in the corresponding category folder
            self.save_resume(file_path, category)

    def read_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def read_doc(self, file_path):
        text = textract.process(file_path)
        return text.decode('utf-8')

    def read_pdf(self, file_path):
        doc = fitz.open(file_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        return '\n'.join(full_text)

    def save_resume(self, file_path, category):
        category_folder = os.path.join(os.getcwd(), category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # Copy the resume to the category folder
        base_filename = os.path.basename(file_path)
        new_path = os.path.join(category_folder, base_filename)
        
        shutil.copy(file_path, new_path)
        messagebox.showinfo("Saved", f"Resume saved to {new_path}.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeClassifierApp(root)
    root.mainloop()

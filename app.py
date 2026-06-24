from fastapi import FastAPI
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()


model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

if torch.backend.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)#lines
    text = re.sub(r"\s+"," ",text) # spaces
    text = re.sub(r"<.*?>"," ",text) # html tags
    text = text.string().lower() # lower case

    return text


def summarize_text(dialogue):
    dialogue= clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
        )
    
    #decode the output

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary 


# api endpoints

@app.get("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_text(dialogue_input.dialogue)
    return {"summary": summary}
    



@app.post("/" ,response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
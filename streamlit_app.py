import os
import re
import fitz  # PyMuPDF
from PIL import Image
import base64
import streamlit as st
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from io import BytesIO

# Azure OpenAI Setup
os.environ["OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_API_KEY"] = "8f74251696ce45698eb95495269f3d8c"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cs-lab-azureopenai-gpt4.openai.azure.com/"
llm = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment="gpt-4o",
    model_name="gpt-4o",
    temperature=0)

#Cache extracted text to avoid reprocessing
@st.cache_data
def extract_text_from_pdf(pdf_path):
    output_folder = 'synoes_images'
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    final_text = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
        img.save(image_path, 'PNG')

        # Skipping the image cleaning for simplicity in this version
        # Encoding image to Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        messages=[
          {"role": "system", "content": "You are a helpful document parsing assistant who is an expert in understanding text. You also have a great understanding of complex document layouts and structures."},
          {"role": "user", "content": [
              {"type": "text", "text": "Read the image given and strictly return only the formatted text in it. Maintain the structure of the content while returning the formatted text. "},
              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
          ]}
        ]
        response = llm.invoke(messages)
        final_text.append(response.content)

    return final_text




# Cache the split sections to avoid rerunning the split process
@st.cache_data
def split_document_by_subfund(text):
    pattern = r'\(“PMF II”\) – .*? \(the “Sub-Fund \w+”\)'
    matches = list(re.finditer(pattern, text))
    sections = []
    for i in range(len(matches)):
        start_index = matches[i].end()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start_index:end_index].strip()
        if section_content:
            sections.append(section_content)
    return sections


def extract_table_from_subfund(subfund_text):
    prompt_template = """ As an intelligent AI Assistant, you have to analyze the Input Data carefully and follow the instructions below:
              - You have to extract the KPIs and their associated values from the Input Data.
              - If you dont found any value associated with any guideline strictly return all the KPIs with 'no value found' and all guidelines
              - Strictly look into each guideline carefully and provide the entities and value only.
              - Strictly The response should be in the form of table only.
              - Don't try to make up an answer if you don't know the answer.
              - Response should contain three columns 'Policy','Description','Guideline' and 'value'
 
    ### Input Data:{input_data} """
    prompt = PromptTemplate(input_variables=["input_data"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(input_data=subfund_text)

    # Split response using '|' and format it into rows
    lines = response.strip().split("\n")
    rows = [line.split("|")[1:-1] for line in lines if "|" in line and len(line.split("|")) > 1]

    # Convert the extracted rows into a DataFrame
    df = pd.DataFrame(rows, columns=["Policy", "Parameter", "Guideline", "Value"])
    df = df[2:].reset_index(drop=True)
    df.index = df.index + 1

    return df

# Function to convert dataframe to excel for download

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:  # Use openpyxl here
        df.to_excel(writer, index=False, sheet_name='SubFund Data')
    output.seek(0)  # Reset the pointer of BytesIO object to the beginning
    return output.getvalue()




#Streamlit App
def main():
    st.title("Investment Policy Statement")

    # Sidebar for file upload and sub-fund selection
    with st.sidebar:
        # Upload PDF File
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Save PDF
            pdf_path = os.path.join("uploaded_pdf.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from PDF
            #st.write("Extracting text from PDF...")
            final_text = extract_text_from_pdf(pdf_path)
            
            # Save the extracted text
            with open("final_text_output.txt", "w") as f:
                f.write("\n".join(final_text))

            # Split document by sub-fund
            #st.write("Splitting document into sub-funds...")
            split_sections = split_document_by_subfund("\n".join(final_text))
            
            # Create Dropdown to Select Sub-Fund
            subfund_options = [f"Sub-fund {i+1}" for i in range(len(split_sections))]
            selected_subfund = st.selectbox("Select Sub-Fund", subfund_options)

            # Use a button to trigger the table extraction after selecting a sub-fund
            extract_table = st.button("Extract Table")
    
    # Main screen: Show table and download option after sub-fund is selected and table is extracted
    if uploaded_file is not None and extract_table:
        selected_index = subfund_options.index(selected_subfund)
        subfund_text = split_sections[selected_index]

        # Extract table from the selected sub-fund
        st.write(f"Extracting table for {selected_subfund}...")
        table_df = extract_table_from_subfund(subfund_text)

        # Display the table as a DataFrame in Streamlit (center)
        st.dataframe(table_df)

        # Convert DataFrame to Excel and allow user to download it
        excel_data = convert_df_to_excel(table_df)
        st.download_button(
            label="Download Table as Excel",
            data=excel_data,
            file_name=f"{selected_subfund}_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()


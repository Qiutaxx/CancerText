import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv  
load_dotenv()

client = OpenAI(
    api_key = os.getenv("QWEN_API_KEY"), 
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

PROMPT_TEMPLATE = """
Role: Expert Molecular Pathologist.
Task: Generate a **high-density biological summary** for patient embedding based on STRICT diagnostic thresholds.

**Diagnostic Rules (Apply these strictly to filter noise):**
1. **mRNA**: High if Z > 1.30; Low if Z < -1.92. (Ignore -1.92 to 1.30).
2. **Methylation**: Hyper-methylated if Z > 1.50; Hypo-methylated if Z < -1.47. (Ignore others).
3. **miRNA**: Upregulated if Z > 1.99; Downregulated if Z < -1.25.
4. **CNV (Genomic Status)**:
   - Value < -1.10 -> **Deletion**
   - -1.10 <= Value < -0.10 -> **Normal (IGNORE)**
   - -0.10 <= Value < 0.90 -> **Gain**
   - 0.90 <= Value < 1.90 -> **Amplification**
   - Value >= 1.90 -> **High-level Amplification**

**Output Constraints:**
1. **Filter**: ONLY report features that meet the criteria above. IGNORE everything else.
2. **Causal Logic**: Connect genomic/epigenetic drivers to transcriptional outcomes (e.g., "High-level Amplification 
drives mRNA overexpression").
3. **No Fluff**: Start directly with the biology. No "The patient shows...".
4. **Length**: STRICTLY under 60 words.

**Desired Output Style**:
"High-level Amplification of [Gene A] drives significant mRNA overexpression. 
[Gene B] promoter hypermethylation mediates epigenetic silencing. 
Deletion of [Gene C] correlates with transcriptional downregulation. 
These features collectively indicate a [Subtype] phenotype with [Pathway] dysregulation."

**Data to Analyze**:
{omics_data_string}

**Output**:
"""


def main():

    input_file = '702_data_string.csv'

    df = pd.read_csv(input_file)
    
    print(f"Successfully read file with {len(df)} rows. Starting analysis...")
    

    analysis_results = []

    for index, row in df.iterrows():
        patient_id = row['PatientID']
        omics_data = row['Input_Data_String'] 
        print(f"Processing row {index + 1}: {patient_id} ...")
        
        try:
            # === Core API Call ===
            completion = client.chat.completions.create(
                model = "qwen3-max",  
                messages = [
                    {"role": "system", "content": PROMPT_TEMPLATE},
                    {"role": "user", "content": f"Here is the omics data row for analysis:\n{omics_data}"}
                ]
            )
            
            analysis_text = completion.choices[0].message.content
            
            analysis_results.append({
                "PatientID": patient_id,
                "Text_Genetate": analysis_text, 
                "Raw_Data": omics_data 
            })
            

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            analysis_results.append({
                "PatientID": patient_id, 
                "Text_Genetate": f"Error: {e}"
            })
        

        time.sleep(0.5)

    output_df = pd.DataFrame(analysis_results)
    output_filename = "Text_node.csv"
    output_df.to_csv(output_filename, index = False, encoding = 'utf-8-sig') 


if __name__ == "__main__":
    main()
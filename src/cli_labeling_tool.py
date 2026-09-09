import pandas as pd
import os

def label_data(csv_path="data/processed/golden_set_unlabeled.csv", output_path="data/processed/golden_set.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # If the user is resuming, we might have already saved to golden_set.csv
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        # Initialize columns if they don't exist
        if 'intent_label' not in df.columns:
            df['intent_label'] = ""
            df['escalate_label'] = ""
            df['escalation_reason'] = ""
            
    # Force string type to prevent pandas FutureWarnings
    df['intent_label'] = df['intent_label'].astype(str)
    df['escalate_label'] = df['escalate_label'].astype(str)
    df['escalation_reason'] = df['escalation_reason'].astype(str)
            
    intents = {
        '1': 'Login/Account Issue',
        '2': 'Audio/Playback Issue',
        '3': 'Subscription/Billing',
        '4': 'Feature Request',
        '5': 'General Inquiry/Other'
    }
    
    unlabeled_idx = df[df['intent_label'].isna() | (df['intent_label'] == "")].index
    
    print(f"Total examples to label: {len(unlabeled_idx)} out of {len(df)}")
    print("Type 'q' at any time to quit and save progress.\n")
    
    for i, idx in enumerate(unlabeled_idx):
        os.system('clear' if os.name == 'posix' else 'cls')
        row = df.loc[idx]
        
        print(f"--- Example {i+1} / {len(unlabeled_idx)} ---")
        print("\nCUSTOMER TWEET:")
        print(f"\"{row['customer_text']}\"")
        
        print("\n---")
        print("Select Intent:")
        for key, val in intents.items():
            print(f"  {key}. {val}")
            
        intent_choice = input("\nIntent (1-5) or 'q' to quit: ").strip().lower()
        if intent_choice == 'q':
            break
            
        # Map choice to string, default to Other if invalid
        intent_str = intents.get(intent_choice, 'General Inquiry/Other')
        
        escalate_choice = input("Escalate this issue to a human? (y/n) [default: n]: ").strip().lower()
        if escalate_choice == 'q':
            break
            
        escalate_bool = "Yes" if escalate_choice == 'y' else "No"
        
        reason_str = ""
        if escalate_bool == "Yes":
            reason_str = input("Reason for escalation: ").strip()
            
        # Update DataFrame
        df.at[idx, 'intent_label'] = intent_str
        df.at[idx, 'escalate_label'] = escalate_bool
        df.at[idx, 'escalation_reason'] = reason_str
        
        # Save incrementally
        df.to_csv(output_path, index=False)
        
    print(f"\nProgress saved to {output_path}!")

if __name__ == "__main__":
    label_data()

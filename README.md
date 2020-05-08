# Autocomplete_Transformer
Using transformer to do autocomplete on requirement document , add AST of code in requirement and knowledge graph as other two inputs of transformer


Hyper:
chunk_len=25
A_end_index=20
B_start_index=5
batch = 128 
nbatches = 4 
transformer_size=12

d_model=d_intermediate=512  # Model.py
d_ff=2048 # Model.py
h=8  # Model.py
lr=0.0002  # Optim.py

1. prepretrain.py
    {audio, video} transformer の fine-tuning

2. pretrain.py
    1. のtransformerの重みを固定にして、共通固有Linearだけ学習

3. train.py
    1. transformerの重みを固定、 
    2. 共通固有Linearの重みを固定 にして、Decoder周り(fusion)学習
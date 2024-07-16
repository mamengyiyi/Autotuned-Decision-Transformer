



for ((seed=1; seed<2; seed+=1))
do
    device="cuda:1"
    env_type=antmaze
    dataset_type=umaze_v2
    hiql_deterministic=False
    hiql_beta=1.0
    update_steps=1000000
    hiql_discount=0.99
    batch_size=256
    way_steps=30
    seq_len=10
    embedding_dim=256
    load_hiql_model="/home/ubuntu/mayi/offline-rl-transformer/dt/checkpoints/HIQL-antmaze-umaze-v2-30-90fe0bc1/checkpoint_999999.pt"
    python -u G-ADT.py --embedding_dim $embedding_dim --seq_len $seq_len --way_steps $way_steps --batch_size $batch_size --hiql_discount $hiql_discount --update_steps $update_steps --hiql_beta $hiql_beta --device $device --train_seed $seed --hiql_deterministic $hiql_deterministic --load_hiql_model $load_hiql_model --config_path ../configs/dt/$env_type/$dataset_type.yaml
done


for ((seed=1; seed<2; seed+=1))
do
    device="cuda:4"
    env_type=antmaze
    dataset_type=large_diverse_v2
    iql_deterministic=False
    iql_beta=6.0
    iql_tau=0.9
    update_steps=1000000
    iql_discount=0.99
    batch_size=256
    load_iql_model="/home/ubuntu/mayi/offline-rl-transformer/dt/checkpoints/IQL-antmaze-large-diverse-v2-b17b79cd/checkpoint_999999.pt"
    python -u V-ADT.py --batch_size $batch_size --iql_discount $iql_discount --update_steps $update_steps --iql_beta $iql_beta --device $device --train_seed $seed --iql_deterministic $iql_deterministic --load_iql_model $load_iql_model --config_path ../configs/dt/$env_type/$dataset_type.yaml
done
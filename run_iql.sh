

# TODO: IQL
for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=hopper
    dataset_type=medium_replay_v2
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done



for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=hopper
    dataset_type=medium_v2
    checkpoints_path='/home/my/offline-rl-transformer/dt/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=hopper
    dataset_type=medium_expert_v2
    checkpoints_path='/home/my/offline-rl-transformer/dt/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done






for ((seed=0; seed<1; seed+=1))
do
    device="cuda:1"
    env_type=antmaze
    dataset_type=umaze_v2
    checkpoints_path='/your/path/to/checkpoints/'
    python -u xql.py --device $device --seed $seed --config_path ../configs/xql/$env_type/$dataset_type.yaml
done

for ((seed=0; seed<3; seed+=1))
do
    device="cuda:1"
    env_type=antmaze
    dataset_type=umaze_diverse_v2
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<3; seed+=1))
do
    device="cuda:2"
    env_type=antmaze
    dataset_type=medium_play_v2
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<3; seed+=1))
do
    device="cuda:3"
    env_type=antmaze
    dataset_type=medium_diverse_v2
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:1"
    env_type=antmaze
    dataset_type=large_play_v2
    checkpoints_path='/home/my/offline-rl-transformer/dt/checkpoints/'
    beta=6.0
    nohup python -u iql.py --beta $beta --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=antmaze
    dataset_type=large_diverse_v2
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u xql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done













for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env=antmaze
    dataset=umaze_v2
    checkpoints_path='/your/path/to/checkpoints/'
    python -u iql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/iql/$env/$dataset.yaml
done

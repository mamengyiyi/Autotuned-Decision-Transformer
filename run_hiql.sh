



for ((seed=0; seed<1; seed+=1))
do
    device="cuda:3"
    env_type=antmaze
    dataset_type=umaze_v2
    checkpoints_path='/your/path/to/checkpoints/'
    way_steps=30
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:4"
    env_type=antmaze
    dataset_type=umaze_diverse_v2
    way_steps=30
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:5"
    env_type=antmaze
    dataset_type=medium_play_v2
    way_steps=30
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:6"
    env_type=antmaze
    dataset_type=medium_diverse_v2
    way_steps=30
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=1; seed<2; seed+=1))
do
    device="cuda:0"
    env_type=antmaze
    dataset_type=large_play_v2
    way_steps=30
    checkpoints_path='/your/path/to/checkpoints/'
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done

for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=antmaze
    dataset_type=large_diverse_v2
    way_steps=30
    checkpoints_path='/home/my/offline-rl-transformer/dt/checkpoints/'
    nohup python -u hiql.py --way_steps $way_steps --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml >/dev/null 2>&1 &
done













for ((seed=0; seed<1; seed+=1))
do
    device="cuda:0"
    env_type=antmaze
    dataset_type=umaze_v2
    checkpoints_path='/your/path/to/checkpoints/'
#    checkpoints_path=None
    python -u hiql.py --device $device --seed $seed --checkpoints_path $checkpoints_path --config_path ../configs/hiql/$env_type/$dataset_type.yaml
done

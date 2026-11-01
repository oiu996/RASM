cd /workspace/RASM
bash improved/improved_try.sh                 # ISTD / full / 300ep
bash improved/improved_try.sh srd mgra 200    # SRD / 只开 MGRA / 200ep
NEPOCH=20 bash improved/improved_try.sh istd full          # 冒烟，先确认能跑通
INIT_CKPT=./log/RASMv2_istd/<run>/models/model_best.pth NEPOCH=600 bash improved/improved_try.sh istd full  # 断点续跑
WEIGHTS=./log/.../model_best.pth bash improved/improved_try.sh istd full   # 只测试不训练
FROM_SCRATCH=1 bash improved/improved_try.sh istd full 300                 # 从零训基线(lr 4e-4)

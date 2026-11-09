# 原始真实RUL>125测试样本的最终预测分布

[Observed] 筛选依据为RUL_FD00x.txt中的未截断真实标签>125，按engine_id对齐先前已保存的原始测试预测。样本数FD00111、FD00257、FD00315、FD00467台发动机；评估目标均已截断为125。比较全数据w48合成训练预测器headline与真实训练基线；不涉及修复置零后的模型。FD002 headline使用7生成种子×5评估种子，其他使用5×5，真实基线各5模型。

[Observed] 直方图汇总各模型对各测试发动机的预测值，并非集成预测；10–90%范围是预测值分位范围，不是均值置信区间。重复模型共用发动机，不是同等数量独立样本。另提供per_engine_summary.csv区分每台发动机的模型间均值/范围。预测输出不截断。

产物：prediction_distribution.png/pdf、distribution_summary.csv、all_model_predictions.csv、per_engine_summary.csv、audit.json。复算：`.venv/bin/python analysis/diagnostics/predictions_for_raw_rul_gt125.py`。

[Observed] FD002/FD004合成训练模型预测均值95.65/93.79，真实训练基线104.51/105.90；目标均为125。它们确实存在相对截断目标的低估，不能解释成与未截断原始真值比较所造成的人为误差。不能单靠该分布做因果归因。

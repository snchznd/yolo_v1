rm ./logs/eval_batch/*.log 2>/dev/null; \
rm ./logs/eval_epoch/*.log 2>/dev/null; \
rm ./logs/events/*.log 2>/dev/null; \
rm ./logs/model/*.pth 2>/dev/null; \
rm ./logs/tensorboard/*events.out.* 2>/dev/null; \
rm ./logs/train_batch/*.log 2>/dev/null; \
rm ./logs/train_epoch/*.log 2>/dev/null; \
echo 'logs cleared'
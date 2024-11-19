import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from toolkit.paths import CONFIG_ROOT

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device_config = self.get_conf('device', 'cpu')
        
        # 判断是否使用多个 GPU
        if isinstance(self.device_config, str) and self.device_config.startswith('cuda'):
            devices = self.device_config.split(',')
            if len(devices) > 1:
                # 多 GPU 情况
                self.distributed = True
                self.local_rank = int(os.getenv('LOCAL_RANK', 0))
                self.device = torch.device(f'cuda:{self.local_rank}')
                self.world_size = len(devices)
                dist.init_process_group(backend='nccl', rank=self.local_rank, world_size=self.world_size)
            else:
                # 单 GPU 情况
                self.distributed = False
                self.device = torch.device(self.device_config)
        else:
            # 默认 CPU 情况
            self.distributed = False
            self.device = torch.device('cpu')

        # 从配置文件加载模型路径或配置参数
        model_path = self.get_conf('model_path', None)  # 假设配置文件中定义了 model_path
        if model_path:
            # 加载预训练模型
            self.model = torch.load(model_path, map_location=self.device)
        else:
            raise ValueError("模型路径未在配置文件中指定，请检查 'model_path' 配置项。")

        # 如果是分布式训练模式，使用 DDP 包装模型
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # 将模型移动到相应设备
        self.model = self.model.to(self.device)

        # 加载扩展的进程
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()

    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.VIPC_PATH        = ''         #path to dataset
#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.DATA_perfetch                          = 8
#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './project_logs'#path to save checkpoints and logs
__C.CONST.DEVICE                                 = '0'
#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.UBCnet                              = edict()
__C.NETWORK.UBCnet.embed_dim                    = 192
__C.NETWORK.UBCnet.depth                        = 6
__C.NETWORK.UBCnet.img_patch_size               = 14
__C.NETWORK.UBCnet.pc_sample_rate               = 0.125
__C.NETWORK.UBCnet.pc_sample_scale              = 2
__C.NETWORK.UBCnet.fuse_layer_num               = 2
__C.NETWORK.shared_encoder                       = edict()
__C.NETWORK.shared_encoder.block_head            = 12
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 16
__C.TRAIN.N_EPOCHS                               = 160
__C.TRAIN.SAVE_FREQ                              = 40
__C.TRAIN.LEARNING_RATE                          = 0.001
__C.TRAIN.LR_MILESTONES                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.LR_DECAY_STEP                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.WARMUP_STEPS                           = 1
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CATE                                   = ''
__C.TRAIN.d_size                                 = 1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.CATE                                    = ''
__C.TEST.BATCH_SIZE                              = 8
#path to pre-trained checkpoints
# __C.CONST.WEIGHTS = r""

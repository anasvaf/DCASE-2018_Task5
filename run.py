from train import train_model, test_model

TRAIN_ITERS = 500000
L_RATE = 1e-3
BATCH_SIZE = 16
LATENT_DIM = [10]
TAU = 0.66
INF_LAYERS = [64, 128]
GEN_LAYERS = [128, 64]
CKPT_PATH = './saved_models/model.ckpt'

def main():
    ops = train_model(TRAIN_ITERS, 10000, L_RATE, BATCH_SIZE, True, 
                      LATENT_DIM, TAU, INF_LAYERS, GEN_LAYERS, CKPT_PATH)
    test_model(ops, 10000, L_RATE, BATCH_SIZE, LATENT_DIM, 
               TAU, INF_LAYERS, GEN_LAYERS, CKPT_PATH)

if __name__ == '__main__':
    main()

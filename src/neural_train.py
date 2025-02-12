"""
Main script for training a neural network to be an SDF or occupancy based network.
"""
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import signal
import sys

from neural_sdf import *
from neural_utils import plot_training_metrics, save_net_object, batch_count_correct
from neural_datasets import SampleDataset, PointCloud

interrupt_flag, sig_handler_set = False, False
def signal_handler(signum, frame):
    global interrupt_flag
    print("\nSignal interrupt detected. Preparing to save the model...")
    interrupt_flag = True

# print(plt.style.available)  # uncomment to view the available plot styles
plt.rcParams['text.usetex'] = False  # tex not necessary here and may cause error if not installed

# Set plot style to seaborn white. If these options do not work, don't set the plot style or select from other
# available plot styles.
try:
    plt.style.use("seaborn-white")
except OSError as e:
    plt.style.use("seaborn-v0_8-white")

set_t = {
    'dtype': torch.float32,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}

def fit_mlp_model(
        net_object: MLP,
        train_loader: DataLoader,
        fit_mode: str,
        epochs: int,
        model_params: dict,
        output_file: str,
        verbose: bool
) -> Tuple[list[float], list[int], list[float], MLP]:
    """
    Given an MLP neurol network and train loader, fit the neural network to the training dataset and record the losses.
    The training heuristics that are returned are:

    * `losses` -- losses per epoch
    * `correct_counts` -- number of predictions that have predicted the correct sign
    * `correct_fracs` -- fraction of predictions that have predicted the correct sign

    :param net_object:       Neural network object to train
    :param train_loader:    Training dataset
    :param fit_mode:        Neural network fitting mode (occupancy or sdf)
    :param epochs:          Number of epochs to run
    :return:                Training heuristics and trained `net_object`
    """

    # global USE_WANDB
    global interrupt_flag, sig_handler_set

    # send to device
    net_object = net_object.to(**set_t)

    # train and record losses
    losses, correct_counts, correct_fracs = [], [], []
    n_correct = 0
    n_total = 0
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y, batch_weight = batch
            batch_x = batch_x.to(**set_t)
            batch_y = batch_y.to(**set_t)
            batch_weight = batch_weight.to(**set_t)
            n_total += len(batch_x)
            curr_epoch_loss = net_object.step(batch_x, batch_y, batch_weight)
            epoch_loss += curr_epoch_loss
            with torch.no_grad():
                correct_count = batch_count_correct(net_object, batch_x, batch_y, fit_mode).item()
                n_correct += correct_count
                correct_counts.append(correct_count)

        # get the current learning rate
        if net_object.scheduler is not None:
            net_object.scheduler.step()
            current_lr = net_object.scheduler.get_last_lr()[0]
        else:
            current_lr = net_object.lr
        # calculate the fraction of correctly predicted signs
        frac_correct= n_correct / n_total
        correct_fracs.append(frac_correct)
        # calculate the epoch loss and update progress bar
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        epoch_details = {
                'Epoch Loss': epoch_loss,
                'Correct Sign': f'{100*frac_correct:.2f}%',
                'Learning Rate': current_lr,
            }
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(epoch_details)
        # if USE_WANDB:
        #     epoch_details.update({'Correct Sign': 100 * frac_correct})
        #     wandb.log(epoch_details)
        if interrupt_flag:
            print(f"\nInterrupt caught during epoch {epoch}. Saving model...")
            save_net_object(net_object, losses, model_params, output_file, verbose=verbose)
            sys.exit(0)  # Exit gracefully

        if sig_handler_set == False:
            signal.signal(signal.SIGINT, signal_handler)
            sig_handler_set = True

    # Reset to default signal handling
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # return metrics and trained network
    return losses, correct_counts, correct_fracs, net_object

def fit_siren_model(
        net_object: Siren,
        train_loader: DataLoader,
        epochs: int,
        model_params: dict,
        output_file: str,
        verbose: bool
) -> Tuple[list[float], Siren]:
    """
    Given a Siren neural network and train loader, fit the neural network to the training dataset and record the losses.

    :param net_object:      Neural network object to train
    :param train_loader:    Training dataset
    :param epochs:          Number of epochs to run
    :return:                Training heuristics and trained `net_object`
    """
    # global USE_WANDB
    global interrupt_flag, sig_handler_set

    # send to device
    net_object = net_object.to(**set_t)

    # train and record losses
    losses = []
    n_total = 0
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss, eik_loss, on_surface_loss, off_surface_loss = [0.0] * 4
        for batch in train_loader:
            # load in batch_data
            inputs, labels = batch
            batch_x = inputs['coords'].squeeze(0)
            batch_normal = labels['normals'].squeeze(0)
            batch_surface_mask = labels['surface_mask'].squeeze(0)
            batch_x = batch_x.to(**set_t)
            batch_normal = batch_normal.to(**set_t)
            batch_surface_mask = batch_surface_mask.to(device=set_t['device'])  # should remain as bool

            n_total += len(batch_x)
            curr_epoch_loss, loss_desc = net_object.step_eikonal(batch_x, batch_normal, batch_surface_mask)

            # update epoch losses
            [curr_eik_loss, curr_on_surface_loss, curr_off_surface_loss] = loss_desc
            epoch_loss += curr_epoch_loss
            eik_loss += curr_eik_loss
            on_surface_loss += curr_on_surface_loss
            off_surface_loss += curr_off_surface_loss

        # get the current learning rate
        current_siren_lr, current_latent_lr = net_object.scheduler_step()
        # calculate the epoch loss and update progress bar
        train_loader_len = len(train_loader)
        epoch_loss /= train_loader_len
        eik_loss /= train_loader_len
        on_surface_loss /= train_loader_len
        off_surface_loss /= train_loader_len
        losses.append(epoch_loss)
        epoch_details = {
                'Epoch Loss': epoch_loss,
                'Eik Loss': eik_loss,
                'On Surface Loss': on_surface_loss,
                'Off Surface Loss': off_surface_loss,
                'Siren Learning Rate': current_siren_lr,
                'Latent Learning Rate': current_latent_lr,
                'Train Loader Length': train_loader_len
            }
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix(epoch_details)
        # if USE_WANDB:
        #     wandb.log(epoch_details)
        if interrupt_flag:
            print(f"\nInterrupt caught during epoch {epoch}. Saving model...")
            save_net_object(net_object, losses, model_params, output_file, verbose=verbose)
            sys.exit(0)  # Exit gracefully

        if sig_handler_set == False:
            signal.signal(signal.SIGINT, signal_handler)
            sig_handler_set = True

    # return metrics and trained network
    return losses, net_object

def train_mlp(args: dict):
    ##  unpack arguments

    # Build arguments
    # program_mode = args["program_mode"]
    input_file = args["input_file"]
    output_file = args["output_file"]
    if input_file is None or output_file is None:
        raise ValueError("input_file and/or output_file is None")
    # network
    input_dim = args["input_dim"]
    activation = args["activation"]
    n_layers = args["n_layers"]
    layer_width = args["layer_width"]
    # positional encoding params
    positional_encoding = args["positional_encoding"]
    positional_count = args["positional_count"]
    positional_pow_start = args["positional_pow_start"]
    positional_prepend = args["positional_prepend"]

    # loss / data
    optimizer = args["optimizer"]
    clip_gradient_norm = args["clip_gradient_norm"]
    fit_mode = args["fit_mode"]
    n_epochs = args["n_epochs"]
    n_samples = args["n_samples"]
    init_scale_factor = args["init_scale_factor"]
    sample_ambient_range = args["sample_ambient_range"]
    sample_weight_beta = args["sample_weight_beta"]
    sample_221 = args["sample_221"]
    show_sample_221 = args["show_sample_221"]
    sdf_max = args["sdf_max"]
    truncate_output = args["truncate_output"]
    # training
    lr = args["lr"]
    batch_size = args["batch_size"]
    lr_decay_every = args["lr_decay_every"]
    lr_decay_frac = args["lr_decay_frac"]
    # general options
    verbose = args["verbose"]
    display_plots = args["display_plots"]

    print(f"Program Configuration: {args}")

    # if USE_WANDB:
    #     if WANDB_GROUP is None:
    #         WANDB_GROUP = program_mode + '_' + wandb.util.generate_id()
    #     uniq_id = WANDB_GROUP.split('_')[-1]
    #     file_name = input_file.split('/')[-1].split('.obj')[0] + '_' + uniq_id
    #
    #     # start a new wandb run to track this script
    #     tags = [fit_mode, program_mode]
    #     if siren_model:
    #         tags += ['siren']
    #     elif positional_encoding:
    #         tags += ['positional_encoding']
    #     wandb.init(
    #         # set the wandb project and name where this run will be logged
    #         project="main_fit_implicit_torch",
    #         name=file_name,
    #         # track hyperparameters and run metadata
    #         config=args_dict,
    #         # set group
    #         group=WANDB_GROUP,
    #         # set tags
    #         tags=tags
    #     )

    # print(f"WANDB ENABLED: {USE_WANDB} | WANDB GROUP: {WANDB_GROUP}")

    # validate some inputs
    if activation not in ['relu', 'elu', 'gelu', 'cos']:
        raise ValueError("unrecognized activation")
    if fit_mode not in ['occupancy', 'sdf']:
        raise ValueError("unrecognized activation")
    if not output_file.endswith('.npz'):
        raise ValueError("output file should end with .npz")

    # initialize the network
    model_params = {
        'input_dim': input_dim,
        'lrate': lr,
        'fit_mode': fit_mode,
        'activation': activation,
        'n_layers': n_layers,
        'layer_width': layer_width,
        'sdf_max': sdf_max,
        'truncate_output': truncate_output,
        'use_positional_encoding': positional_encoding,
        'positional_count': positional_count,
        'positional_power_start': positional_pow_start,
        'positional_prepend': positional_prepend,
        'optimizer': optimizer,
        'with_shift': True,
        'step_size': lr_decay_every,
        'gamma': lr_decay_frac,
        'clip_gradient_norm': clip_gradient_norm
    }
    net_object = MLP(**model_params)

    # initialize the dataset
    dataset_pararms = {
        'input_file': input_file,
        'fit_mode': fit_mode,
        'n_samples': n_samples,
        'sample_weight_beta': sample_weight_beta,
        'sample_ambient_range': sample_ambient_range,
        'sample_221': sample_221,
        'show_sample_221': show_sample_221,
        'sdf_max': sdf_max,
        'truncate_outputs': truncate_output,
        'init_scale_factor': init_scale_factor,
        'verbose': verbose
    }
    train_dataset = SampleDataset(**dataset_pararms)
    batch_size = min(batch_size, len(train_dataset))
    print(f"Batch Size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # train the neural network
    losses, correct_counts, correct_fracs, net_object = fit_mlp_model(net_object, train_loader, fit_mode, n_epochs,
                                                                     model_params, output_file, verbose)

    # save the model
    save_net_object(net_object, losses, model_params, output_file, verbose=verbose)

    # if USE_WANDB:
    #     wandb.finish()

def train_siren(args: dict):
    ##  unpack arguments

    # Build arguments
    input_file = args["input_file"]
    output_file = args["output_file"]
    if input_file is None or output_file is None:
        raise ValueError("input_file and/or output_file is None")
    # network
    input_dim = args["input_dim"]
    n_layers = args["n_layers"]
    layer_width = args["layer_width"]
    clip_gradient_norm = args["clip_gradient_norm"]
    # siren params
    siren_latent_dim = args["siren_latent_dim"]
    siren_first_omega_0 = args["siren_first_omega_0"]
    siren_hidden_omega_0 = args["siren_hidden_omega_0"]
    siren_c1 = args["siren_c1"]
    siren_c2 = args["siren_c2"]
    siren_c3 = args["siren_c3"]

    # loss / data
    fit_mode = args["fit_mode"]
    n_epochs = args["n_epochs"]
    init_scale_factor = args["init_scale_factor"]
    # training
    siren_lr = args["lr"]
    final_siren_lr = args["final_siren_lr"]
    latent_lr = args["latent_lr"]
    final_latent_lr = args["final_latent_lr"]
    scheduler_type = args["scheduler_type"]
    batch_size = args["batch_size"]
    lr_decay_every = args["lr_decay_every"]
    lr_decay_frac = args["lr_decay_frac"]
    # general options
    verbose = args["verbose"]
    display_plots = args["display_plots"]

    print(f"Program Configuration: {args}")

    # if enabled, initializes wandb and prints a url to view the training progress online at wandb.ai
    # if USE_WANDB:
    #     if WANDB_GROUP is None:
    #         WANDB_GROUP = 'manuscript_' + wandb.util.generate_id()
    #     uniq_id = WANDB_GROUP.split('_')[-1]
    #     file_name = input_file.split('/')[-1].split('.obj')[0] + '_' + uniq_id
    #
    #     # start a new wandb run to track this script
    #     tags = [fit_mode, 'siren_eik']
    #     if siren_latent_dim > 0:
    #         tags.append('latent_modulation')
    #     wandb.init(
    #         # set the wandb project and name where this run will be logged
    #         project="main_fit_implicit_point_cloud",
    #         name=file_name,
    #         # track hyperparameters and run metadata
    #         config=args_dict,
    #         # set group
    #         group=WANDB_GROUP,
    #         # set tags
    #         tags=tags
    #     )

    # print(f"WANDB ENABLED: {USE_WANDB} | WANDB GROUP: {WANDB_GROUP}")

    # build the neural network with the specified configuration
    model_params = {
        'in_features': input_dim,
        'hidden_features': layer_width,
        'hidden_layers': n_layers,
        'out_features': 1,
        'num_epochs': n_epochs,
        'siren_lrate': siren_lr,
        'final_siren_lrate': final_siren_lr,
        'latent_lrate': latent_lr,
        'final_latent_lrate': final_latent_lr,
        'scheduler_type': scheduler_type,
        'first_omega_0': siren_first_omega_0,
        'hidden_omega_0': siren_hidden_omega_0,
        'latent_dim': siren_latent_dim,
        'step_size': lr_decay_every,
        'gamma': lr_decay_frac,
        'c1': siren_c1,
        'c2': siren_c2,
        'c3': siren_c3,
        'clip_gradient_norm': clip_gradient_norm
    }
    net_object = Siren(**model_params)

    # load the dataset
    # TODO: This PointCloud dataset has not been modified to handle 2D images as of yet
    sdf_dataset = PointCloud(input_file, on_surface_points=batch_size)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1)

    # train the neural network
    losses, net_object = fit_siren_model(net_object, dataloader, n_epochs, model_params, output_file, verbose)

    # save the model
    save_net_object(net_object, losses, model_params, output_file)

    # if USE_WANDB:
    #     wandb.finish()

def main(args: dict):
    """
    Main function for training an MLP or Siren network for an SDF/occupancy based function.
    :param args:
    :return:
    """
    # TODO: Ideally, nglod should be included as an option here as well.
    model_type = args.pop('model_type').lower()
    if model_type == 'mlp':
        train_fn = train_mlp
    elif model_type == 'siren':
        train_fn = train_siren
    else:
        raise ValueError('Unknown model type')

    print(f"Torch Settings: {set_t}")

    train_fn(args)

def parse_args() -> dict:
    """
    More explicitly organizes the available arguments available in this program.
    :return:
    """
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="The input dataset to use for training.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Name of the file to save the model and plots.")

    # network
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of model to use (MLP or Siren).")  # MLP or Siren
    parser.add_argument("--input_dim", type=int, default=3,
                        help="Input dimension of the neural network model.")
    parser.add_argument("--activation", type=str, default='relu',
                        help="Type of activation function to use at each intermediate layer.")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of layers to use for the network.")
    parser.add_argument("--layer_width", type=int, default=32,
                        help="Number of neurons per layer.")
    parser.add_argument("--truncate_output", action='store_true',
                        help="Truncates the ground-truth distances to sdf_max and ensures the output of the MLP "
                             "is in the range [-1, 1]. Otherwise the ground-truth distances are preserved (but still "
                             "weighted with respect to sdf_max), and the 'tanh' activation is not used at the output.")
    parser.add_argument("--clip_gradient_norm", type=float,
                        help="Maximum norm of gradients to clip to for aid with training stability.")
    #positional arguments
    parser.add_argument("--positional_encoding", action='store_true',
                        help="For an MLP network, prepends a positional encoding layer.")
    parser.add_argument("--positional_count", type=int, default=10,
                        help="If positional encoding is enabled, sets the number of outputs.")
    parser.add_argument("--positional_pow_start", type=int, default=-3,
                        help="If positional encoding is enabled, defines sinusoidal frequency to start with.")
    parser.add_argument("--positional_prepend", action='store_true',
                        help="If positional encoding is enabled, prepends the network input to the output of the "
                             "positional encoding layer.")
    # siren arguments
    parser.add_argument("--siren_latent_dim", type=int, default=0,
                        help="The dimension of the latent variabel for a Siren network that is modulated.")
    parser.add_argument("--siren_first_omega_0", type=int, default=30,
                        help="Sinusoidal frequency of the first Sine layer of a Siren network.")
    parser.add_argument("--siren_hidden_omega_0", type=int, default=30,
                        help="Sinusoidal frequency of the hidden layers of a Siren network.")
    parser.add_argument("--siren_c1", type=float, default=5e1)
    parser.add_argument("--siren_c2", type=float, default=3e3)
    parser.add_argument("--siren_c3", type=float, default=1e2)

    # loss / data
    parser.add_argument("--fit_mode", type=str, default='sdf', choices=['sdf', 'occupancy'],
                        help="Type of function to fit. The neural network should be trained to be sdf "
                             "(signed distance function) or occupancy.")
    parser.add_argument("--n_samples", type=int, default=1000000,
                        help="Number of samples to use per epoch.")
    parser.add_argument("--sample_ambient_range", type=float, default=1.25)
    parser.add_argument("--sample_weight_beta", type=float, default=20.)
    parser.add_argument("--sample_221", action='store_true')
    parser.add_argument('--show_sample_221', action='store_true')
    parser.add_argument("--sdf_max", type=float, default=0.1)

    # training
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size per epoch.")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd', 'lbfgs'],
                        help="Optimizer to use for training.")
    parser.add_argument("--init_scale_factor", type=int, default=2,
                        help="For loading a 2D png image to use an SDF, the original image may not produce enough "
                             "samples. In this case, the image will iteratively get refactored until the number of "
                             "samples is at least n_samples. This process can be redundant if the user already knows "
                             "the scale factor that should be used.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Main network learning rate.")
    parser.add_argument("--final_siren_lr", type=float, default=None)
    parser.add_argument("--latent_lr", type=float, default=1e-2,
                        help="Learning rate to use for the latent variable of a Siren network that is modulated.")
    parser.add_argument("--final_latent_lr", type=float, default=None)
    parser.add_argument("--lr_decay_every", type=int, default=None)
    parser.add_argument("--lr_decay_frac", type=float, default=None)
    parser.add_argument('--scheduler_type', type=str, default='none')

    # general options
    parser.add_argument("--verbose", action='store_true',
                        help="If true, prints additional information during training.")
    parser.add_argument("--display_plots", action='store_true')
    parser.add_argument('--check_csv_table', type=str, default=None)

    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm.auto import tqdm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


# ==================================
# 1. CONFIGURATION
# ==================================
img_size = 128
batch_size = 256
learning_rate = 4e-4
latent_dim = 256
n_clusters = 9  # Numero di cluster per il modello CVAE
epoch_Vade_start = 50  # Inizia a usare VADE dopo 50 epoche
num_epochs = 150  # Puoi modificare il numero di epoche per ogni sessione
dataset_path = "./landscape_ridotto/"
model_path = "./"  # Percorso per salvare il modello
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seed per la riproducibilità
torch.manual_seed(300)

# ==================================
# 2. DATASET E DATA LOADER
# ==================================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print(f"{len(dataset)} immagini caricate.")

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==================================
# 3. VADE MODEL DEFINITION
# ==================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 1x1 spatial dimension, necessary for SE block beacause it reduces the feature map to a single value per channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VADE(nn.Module):
    """
    Variational Deep Embedding (VaDE) model with Squeeze-and-Excitation (SE) blocks for image data.

    This class implements a deep generative clustering model that combines a Variational Autoencoder (VAE)
    with a Gaussian Mixture Model (GMM) prior in the latent space. The encoder and decoder are built using
    convolutional and transposed convolutional layers, respectively, with SE blocks to enhance feature
    representation. The model is suitable for unsupervised clustering and generative modeling of images.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        img_size (int): Height/width of the (square) input images.
        n_clusters (int): Number of clusters (components) in the GMM prior.

    Attributes:
        pi_logits (nn.Parameter): Logits for the cluster prior probabilities (π).
        mu_prior (nn.Parameter): Means of the Gaussian components in the latent space.
        log_var_prior (nn.Parameter): Log-variances of the Gaussian components in the latent space.
        encoder (nn.Sequential): Convolutional encoder with SE blocks.
        fc_mu (nn.Linear): Fully connected layer to compute latent mean.
        fc_logvar (nn.Linear): Fully connected layer to compute latent log-variance.
        fc_decode (nn.Linear): Fully connected layer to project latent vector to decoder input.
        decoder (nn.Sequential): Transposed convolutional decoder with SE blocks.

    Methods:
        encode(x):
            Encodes input images into latent mean and log-variance.
            Args:
                x (Tensor): Input image batch of shape (B, 3, img_size, img_size).
            Returns:
                mu (Tensor): Latent mean of shape (B, latent_dim).
                logvar (Tensor): Latent log-variance of shape (B, latent_dim).

        reparameterize(mu, logvar):
            Samples latent variable z using the reparameterization trick.
            Args:
                mu (Tensor): Latent mean.
                logvar (Tensor): Latent log-variance.
            Returns:
                z (Tensor): Sampled latent variable.

        decode(z):
            Decodes latent variable z back to image space.
            Args:
                z (Tensor): Latent variable of shape (B, latent_dim).
            Returns:
                recon_x (Tensor): Reconstructed images of shape (B, 3, img_size, img_size).

        forward(x):
            Full forward pass: encodes input, samples latent, decodes output.
            Args:
                x (Tensor): Input image batch.
            Returns:
                recon_x (Tensor): Reconstructed images.
                mu (Tensor): Latent mean.
                logvar (Tensor): Latent log-variance.

    Notes:
        - The encoder downsamples the input image by a factor of 32 using 5 convolutional layers with stride=2.
        - The decoder upsamples the latent representation back to the original image size using transposed convolutions.
        - Squeeze-and-Excitation (SE) blocks are used after each convolutional layer to recalibrate channel-wise features.
        - The model assumes input images are RGB and square-shaped.
    """
    def __init__(self, latent_dim, img_size, n_clusters):
        super(VADE, self).__init__()
        self.pi_logits = nn.Parameter(torch.zeros(n_clusters))
        self.mu_prior = Parameter(torch.zeros(n_clusters, latent_dim))
        self.log_var_prior = Parameter(torch.randn(n_clusters, latent_dim))
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Calculate the size after convolutions
        conv_output_size = img_size // 32  # Now 5 Conv layers with stride=2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2,
                      padding=2),   # (img_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),  # (img_size/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.Conv2d(128, 256, kernel_size=3, stride=2,
                      padding=1),  # (img_size/8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock(256),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.Conv2d(256, 512, kernel_size=3, stride=2,
                      padding=1),  # (img_size/16)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SEBlock(512),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2,
                      padding=1),  # (img_size/32)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SEBlock(1024),  # Squeeze-and-Excitation block
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(1024 * conv_output_size *
                               conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(
            1024 * conv_output_size * conv_output_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(
            latent_dim, 1024 * conv_output_size * conv_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # (img_size/16)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SEBlock(512),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),   # (img_size/8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock(256),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),   # (img_size/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),    # (img_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),  # Squeeze-and-Excitation block
            nn.Dropout(p=0.05),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1),      # (img_size)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        conv_output_size = self.img_size // 32
        h = h.view(-1, 1024, conv_output_size, conv_output_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==================================
# 4. MODEL INITIALIZATION
# ==================================


# Example usage
model = VADE(latent_dim=latent_dim, img_size=img_size,
              n_clusters=n_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

# ==================================
# 5. SUPPORT FUNCTIONS
# ==================================

def compute_gamma(z, pi_prior, mu_prior, log_var_prior):
    """
    Computes the responsibilities gamma (q(c|x)) in a numerically stable way.
    This implements Equation 10 from the VaDE paper.
    Uses log-space operations to avoid underflow/overflow.

    Args:
        z (torch.Tensor): Latent vectors sampled from q(z|x), shape (batch_size, latent_dim).
        pi_prior (torch.Tensor): Cluster weights (prior p(c)), shape (n_clusters,).
        mu_prior (torch.Tensor): Cluster means (prior p(z|c)), shape (n_clusters, latent_dim).
        log_var_prior (torch.Tensor): Cluster log-variances (prior p(z|c)), shape (n_clusters, latent_dim).

    Returns:
        torch.Tensor: Responsibilities gamma, shape (batch_size, n_clusters).
    """

    z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
    mu_prior_expanded = mu_prior.unsqueeze(0) # [1, n_clusters, latent_dim]
    log_var_prior_expanded = log_var_prior.unsqueeze(0) # [1, n_clusters, latent_dim]

    # 2. Compute the log probability density of the Gaussian: log p(z|c)
    # The formula is -0.5 * [d*log(2π) + log_var + (z-mu)²/var]
    # We ignore the log(2π) constant because it cancels out in the next step (softmax).
    log_p_z_given_c = -0.5 * torch.sum(
        log_var_prior_expanded +
        (z_expanded - mu_prior_expanded).pow(2) /
        torch.exp(log_var_prior_expanded),
        dim=2  # Sum over latent dimension to get one value per cluster
    )

    # Compute the unnormalized log posterior: log( p(c) * p(z|c) )
    # Add a small epsilon to pi_prior to avoid log(0)
    log_gamma_unnormalized = torch.log(
        pi_prior.unsqueeze(0) + 1e-10) + log_p_z_given_c

    # Normalize using softmax to get the final responsibilities (gamma)
    # torch.softmax computes exp(x) / sum(exp(x)) in a numerically stable way
    gamma = torch.softmax(log_gamma_unnormalized, dim=1)

    return gamma


def collect_all_mu(model, data_loader, device):
    """
    Collects all latent mean vectors (mu) from the dataset using the provided data_loader.
    Returns a tensor of shape [num_samples, latent_dim].
    """
    model.eval()
    mus = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            mu_batch = model.fc_mu(model.encoder(x_batch))
            mus.append(mu_batch.cpu())
    return torch.cat(mus, dim=0)


def fit_gmm_and_set_priors(model, all_mu, n_clusters, device):
    """
    Runs GMM clustering on all latent mean vectors and updates the model's prior parameters.
    """
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type='diag', max_iter=1000)
    gmm.fit(all_mu.numpy())
    initial_pi = torch.tensor(gmm.weights_, device=device)
    model.pi_logits.data = torch.log(initial_pi + 1e-10)
    model.mu_prior.data = torch.tensor(gmm.means_, device=device)
    model.log_var_prior.data = torch.tensor(
        np.log(gmm.covariances_ + 1e-10), device=device)

    # Stampa i logits (valori non vincolati)
    print("GMM logits updated:", model.pi_logits.data)

    # Oppure, ancora meglio, stampa le probabilità risultanti
    print("GMM initial probabilities:",
          torch.softmax(model.pi_logits, dim=0).data)

    print("GMM mu_prior updated:", model.mu_prior.data)
    print("GMM log_var_prior updated:", model.log_var_prior.data)

# ==================================
# 6. LOSS FUNCTION
# ==================================
def vade_loss(x, x_hat, mu, log_var, z, model, epochs, epoch_Vade_start):
    """
    Computes the loss for the VaDE model.

    Args:
        x (Tensor): Original input images.
        x_hat (Tensor): Reconstructed images from the decoder.
        mu (Tensor): Latent mean from the encoder.
        log_var (Tensor): Latent log-variance from the encoder.
        z (Tensor): Sampled latent variable.
        model (VADE): The VaDE model instance.
        epochs (int): Current training epoch.
        epoch_Vade_start (int): Epoch at which to start using the VaDE loss.

    Returns:
        Tensor: The computed loss value.
    """
    # Before epoch_Vade_start, use only MSE reconstruction loss (pure autoencoder pretraining)
    if epochs <= epoch_Vade_start:
        mse_loss = F.mse_loss(x_hat, x, reduction='sum')
        loss = mse_loss / x.size(0)  # Normalize by batch size
        return loss
    else:
        # After epoch_Vade_start, use full VaDE loss (ELBO)
        pi_prior = torch.softmax(model.pi_logits, dim=0)
        mu_prior = model.mu_prior
        log_var_prior = model.log_var_prior
        gamma = compute_gamma(z, pi_prior, mu_prior, log_var_prior)

        # 1. Reconstruction loss (MSE between input and output)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')

        # 2. KL divergence term (regularization)
        # This term encourages the latent distribution to match the GMM prior
        # KL(q(z,c|x) || p(z,c)) is expanded as in the VaDE paper (Equation 9)

        # E[log p(z|c)] term: expectation of log-likelihood under the GMM prior
        log_p_z_c = -0.5 * torch.sum(
            gamma * torch.sum(
                log_var_prior.unsqueeze(0) +
                (log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - mu_prior).pow(2)) /
                log_var_prior.exp(), dim=2)
        )

        # E[log p(c)]: expectation of the log prior cluster probabilities
        log_p_c = torch.sum(
            gamma * torch.log(pi_prior.unsqueeze(0) + 1e-10), dim=1)

        # E[log q(z|x)]: entropy of the approximate posterior
        log_q_z_x = -0.5 * torch.sum(1 + log_var, dim=1)

        # E[log q(c|x)]: entropy of the cluster assignment probabilities
        log_q_c_x = torch.sum(gamma * torch.log(gamma + 1e-10), dim=1)

        # Combine all terms to compute the KL divergence
        kl_divergence = log_p_z_c + log_p_c.sum() - log_q_z_x.sum() - log_q_c_x.sum()

        # The final loss is negative ELBO: reconstruction - KL divergence
        # The KL term is scaled by 0.1 for stability (can be tuned)
        loss = reconstruction_loss - 0.1 * kl_divergence

        # Normalize by batch size
        loss /= x.size(0)

        # Print loss components for monitoring
        print(f"[Epoch {epochs}] Total Loss: {loss.item()}")
        print(f"[Epoch {epochs}] Reconstruction Loss: {reconstruction_loss.item() / x.size(0)}")
        print(f"[Epoch {epochs}] KL Divergence: {kl_divergence.item() / x.size(0)}")

        return loss

# ===========================
# 7. TRAINING LOOP
# ===========================

for epoch in range(num_epochs):
    model.train()
    # Progress bar for each epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for x, _ in pbar:
        x = x.to(device)
        optimizer.zero_grad()
        # Forward pass: get reconstruction, latent mean and logvar
        x_hat, mu, log_var = model(x)
        # Sample latent variable z using reparameterization trick
        z = model.reparameterize(mu, log_var)
        # Compute the loss (MSE or full VaDE loss depending on epoch)
        loss = vade_loss(x, x_hat, mu, log_var, z,
                         model, epoch, epoch_Vade_start)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    # At the specified epoch, update GMM priors using the latent means
    if epoch == epoch_Vade_start:
        print("Updating priors with GMM...")
        all_mu = collect_all_mu(model, train_loader, device)
        fit_gmm_and_set_priors(model, all_mu, n_clusters, device)
        print("Priors updated with GMM.")
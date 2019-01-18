"""
Reference
https://qiita.com/shinmura0/items/811d01384e20bfd1e035
https://confit.atlas.jp/guide/event-img/jsai2018/2A1-03/public/pdf?type=in
https://gist.github.com/colspan/bb029025881ddcdce9f70838aff4aa82
"""

import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class CVAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, c_ch, n_ch, z_dim, latent_dim, C=1.0, k=1, wscale=0.02):
        """
        Args:
            args below are for loss function
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """
        """
        !! Network Requirment!!
        Source : https://confit.atlas.jp/guide/event-img/jsai2018/2A1-03/public/pdf?type=in

        memo:
        Convolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1)
        
        Init channel = Nc
        Dimension pf z = Nz
        
        In Layer n
        
        # ksize(Size of kernel) = (4, 4)
        # stride = 2
        # channel = Nc * 2**(n-1)
        # output = batchnormalization
        # activation = ReLU

        Output Layer
        
        # unit = 2 * Nz
        # # unit of mu = Nz
        # # unit of ln_sigma_2 = Nz

        """
        w = chainer.initializers.Normal(scale=wscale)
        super(CVAE, self).__init__(
            e_c0=L.Convolution2D(c_ch * 1, n_ch * 1, 4, 2, 1, initialW=w),
            e_c1=L.Convolution2D(n_ch * 1, n_ch * 2, 4, 2, 1, initialW=w),
            e_c2=L.Convolution2D(n_ch * 2, n_ch * 4, 4, 2, 1, initialW=w),
            e_c3=L.Convolution2D(n_ch * 4, n_ch * 8, 4, 2, 1, initialW=w),
            e_bn0=L.BatchNormalization(n_ch, use_gamma=False),
            e_bn1=L.BatchNormalization(n_ch * 2, use_gamma=False),
            e_bn2=L.BatchNormalization(n_ch * 4, use_gamma=False),
            e_bn3=L.BatchNormalization(n_ch * 8, use_gamma=False),
            z_mu=L.Linear(n_ch * 8, latent_dim),
            z_ln_var=L.Linear(n_ch * 8, latent_dim),
            d_c0=L.Linear(n_ch*8),
            d_c1=L.Deconvolution2D(n_ch * 8, n_ch * 4, 4, 2, 1, initialW=w),
            d_c2=L.Deconvolution2D(n_ch * 4, n_ch * 2, 4, 2, 1, initialW=w),
            d_c3=L.Deconvolution2D(n_ch * 2, n_ch * 1, 4, 2, 1, initialW=w),
            d_c4=L.Deconvolution2D(n_ch * 1, c_ch * 1, 4, 2, 1, initialW=w),
            d_x1=L.Deconvolution2D(n_ch, c_ch, 4, 2, 1, initialW=w),
            d_x2=L.Deconvolution2D(n_ch, c_ch, 4, 2, 1, initialW=w),
            d_bn0=L.BatchNormalization(n_ch * 4, use_gamma=False),
            d_bn1=L.BatchNormalization(n_ch * 2, use_gamma=False),
            d_bn2=L.BatchNormalization(n_ch * 1, use_gamma=False),
            d_bn3=L.BatchNormalization(c_ch * 1, use_gamma=False),
            d_bn4=L.BatchNormalization(c_ch * 1, use_gamma=False),
            d_bnx1=L.BatchNormalization(c_ch, use_gamma=False),
            d_bnx2=L.BatchNormalization(c_ch, use_gamma=False),
        )

        self.C = C
        self.k = k

    def __call__(self, x):
        """AutoEncoder"""
        mu, ln_var = self.encode(x)
        batchsize = len(mu.data)
        # reconstruction loss
        z = F.gaussian(mu, ln_var)
        outputs_mu, outputs_sigma_2 = self.decode(z)
        
        a_vae_loss = F.log(2 * 3.14 * F.flatten(outputs_sigma_2))
        a_vae_loss = 0.5 * F.sum(a_vae_loss)
        
        m_vae_loss = (F.flatten(x) - F.flatten(outputs_mu))**2 / F.flatten(outputs_sigma_2)
        m_vae_loss = 0.5 * F.sum(m_vae_loss)

        """
        kl_loss = 1 + ln_var - F.square(mu) - F.exp(ln_var)
        kl_loss = F.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        """
        
        d_vae_loss = F.gaussian_kl_divergence(mu, ln_var)
        
        loss = F.mean(d_vae_loss + m_vae_loss + a_vae_loss)
        
        chainer.report({'loss': loss}, self)
        return loss

    def encode(self, x):
        # print("=== encoder ===")
        # print(x.shape)
        h = F.leaky_relu(self.e_bn0(self.e_c0(x)), slope=0.2)
        # print(h.shape)
        h = F.leaky_relu(self.e_bn1(self.e_c1(h)), slope=0.2)
        # print(h.shape)
        h = F.leaky_relu(self.e_bn2(self.e_c2(h)), slope=0.2)
        # print(h.shape)
        h = F.leaky_relu(self.e_bn3(self.e_c3(h)), slope=0.2)
        # print(h.shape)
        mu = self.z_mu(h)
        ln_var = self.z_ln_var(h)
        # print(mu.shape)
        return mu, ln_var

    def decode(self, z):
        # print("=== decoder ===")
        # print(z.shape)
        h = F.relu(self.d_c0(z))
        # print(h.shape)
        h = F.reshape(h, (h.shape[0], 128, 1, 1))
        # print(h.shape)
        h = F.relu(self.d_bn0(self.d_c1(h)))
        # print(h.shape)
        h = F.relu(self.d_bn1(self.d_c2(h)))
        # print(h.shape)
        h = F.relu(self.d_bn2(self.d_c3(h)))
        # print(h.shape)
        x1 = F.sigmoid(self.d_bnx1(self.d_x1(h)))
        x2 = F.sigmoid(self.d_bnx2(self.d_x2(h)))
        # print(f"{x1.shape}, {x2.shape}")
        return x1, x2

    def get_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """

    def lf(self, x):
        """AutoEncoder"""
        mu, ln_var = self.encode(x)
        batchsize = len(mu.data)
        # reconstruction loss
        z = F.gaussian(mu, ln_var)
        outputs_mu, outputs_sigma_2 = self.decode(z)
        
        m_vae_loss = (F.flatten(x) - F.flatten(outputs_mu))**2 / F.flatten(outputs_sigma_2)
        m_vae_loss = 0.5 * F.sum(m_vae_loss)

        a_vae_loss = F.log(2 * 3.14 * F.flatten(outputs_sigma_2))
        a_vae_loss = 0.5 * F.sum(a_vae_loss)

        """
        kl_loss = 1 + ln_var - F.square(mu) - F.exp(ln_var)
        kl_loss = F.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        """

        d_vae_loss = F.gaussian_kl_divergence(mu, ln_var)

        self.loss = F.mean(d_vae_loss + m_vae_loss + a_vae_loss)

        return self.loss

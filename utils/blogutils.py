import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from IPython.core.pylabtools import print_figure
from base64 import b64encode
from cycler import cycler
from io import BytesIO
import re
import requests
import PIL
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

plt.rcParams['figure.dpi'] = 300

# -------------------------------------------------------------------
# NOTEBOOK STYLING
# -------------------------------------------------------------------

class col:
    FG = '#E6F6FE'
    BG = '#FFFFFF'
    PRIMARY = '#BF1616'
    SECONDARY = '#615F5C'
    NEUTRAL_LIGHTER = '#F5F5F5'
    TERTIARY = '#F6F7F7'
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    GRAY = '#AAB9C3'
    PINK = '#F68DA6'
    RED = '#CF2E2E'
    ORANGE = '#FF6801'
    YELLOW = '#FCB900'
    TURQUOISE = '#7BDCB5'
    GREEN = '#01D184'
    SKY_BLUE = '#8ED1FC'
    BLUE = '#0693E3'
    PURPLE = '#9B50E1'

def init_theme(_sns):
    _sns.set_theme(style='darkgrid')
    _sns.set_style('darkgrid', {'axes.facecolor': col.NEUTRAL_LIGHTER })
    _sns.set_palette("Paired")    

    
# -------------------------------------------------------------------
# RESULT PLOTTING UTILS
# -------------------------------------------------------------------

def html_df(df, fignum, figcaption):
    '''Renders as pandas dataframe as a figure with a caption.
    '''
    df_html = df.to_html()
    html = '''
        <figure
            class="nb-generated-diagram"
            align="center"
            style="display:flex; align-items:center; flex-flow:column;"
        >
            {}
            <figcaption>Figure {}: {}</figcaption>
        </figure>
    '''.format(df_html, fignum, figcaption)
    return HTML(html)

def html_fig(fig, fignum, figcaption, source='TeaochaDesign'):
    '''Takes a matplotlib figure and turns it into an HTML
    figure instead.
    '''
    fig_b64 = b64encode(print_figure(fig)).decode("utf-8")
    img_data = f'data:image/png;base64,{fig_b64}'
    if source == '' or source == None:
        source = ''
    else:
        source = f' (Source: {source})'
    html = '''
        <figure class="nb-generated-diagram" align="center">
            <img src="{}">
            <figcaption>Figure {}: {}{}</figcaption>
        </figure>
    '''.format(img_data, fignum, figcaption, source)
    plt.close();
    return HTML(html)

def plot_samples(sample_generator):
    '''Generates a set of samples and plots them in a grid.
    '''
    fig, axes = plt.subplots(2, 4, figsize=(8, 3))
    for row in range(2):
        for col in range(4):
            axes[row][col].axis('off')
            axes[row][col].imshow(next(sample_generator).convert('L'), cmap='Blues_r');
    return fig
    
def plot_experiment_results(df):
    '''Plots results from a wavelet comparision experiment.
    '''
    fig = plt.figure(figsize=(10, 10), constrained_layout=True);
    gs = fig.add_gridspec(3, 2);

    ax0 = fig.add_subplot(gs[0,:]);
    sns.stripplot(ax=ax0, x='wavelet_fam', y='total_score', data=df, hue='remove_levels');
    ax0.set_xlabel('');
    ax0.set_ylabel('Total Loss');
    ax0.legend(title='No. Levels Removed');

    ax10 = fig.add_subplot(gs[1:, 0]);
    sns.stripplot(ax=ax10, x='wavelet_fam', y='compression_score', data=df, hue='remove_levels');
    ax10.set_xlabel('');
    ax10.set_ylabel('Compression Loss');
    ax10.legend(title='No. Levels Removed');

    ax11 = fig.add_subplot(gs[1:, 1]);
    sns.stripplot(ax=ax11, x='wavelet_fam', y='reconstruction_score', data=df, hue='remove_levels');
    ax11.set_xlabel('');
    ax11.set_ylabel('Reconstruction Loss');
    ax11.legend(title='No. Levels Removed');
    
    return fig
    
def calculate_anova(df):
    '''Performs a type-2 ANOVA on the results of a wavelet experiment.
    '''
    anova_model = ols('total_score ~ C(wavelet_fam)', data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    return anova_table

# -------------------------------------------------------------------
# DATA LOADING/MUNGING
# -------------------------------------------------------------------    

def urlimg(url):
    '''
    Pulls an image from the url and decodes it into an image
    object using pillow (because plt.imread is deprecated).
    '''
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)
    
def image_bootstrapper(images_dir: str, x_crop: float=0.5, y_crop: float=0.5):
    '''Given a directory containing images, returns a generator for
    subsamples of the images in that directory.
    
    Params:
        images_dir:
            The directory containing the images
        x_crop:
            The ratio of the image to randomly crop in the x axis
        y_crop:
            The ratio of the image to randomly crop in the y axis
    '''
    image_paths = [
        img_path for img_path in os.listdir(images_dir)
        if re.search(r'\.(png|jpg|jpeg|bmp)$', img_path)
    ]
    images = [
        PIL.Image.open(os.path.join(images_dir, image_path))
        for image_path in image_paths
    ]
    
    while True:
        img = images[np.random.randint(len(images))]
        crop_width = np.math.floor(img.width * x_crop)
        crop_max_x = img.width - crop_width
        crop_x = np.random.randint(crop_max_x) if crop_max_x > 0 else 0
        crop_height = np.math.floor(img.height * y_crop)
        crop_max_y = img.height - crop_height
        crop_y = np.random.randint(crop_max_y) if crop_max_y > 0 else 0
        
        cropped = img.crop((
            crop_x,
            crop_y,
            crop_x + crop_width,
            crop_y + crop_height
        ))
        yield cropped

def kl(P,Q):
    """KL-Divergence of Q with respect to P.
    """
    # Epsilon is used here to avoid conditional code for
    # checking that neither P nor Q is equal to 0.
    epsilon = 0.00001
    P = P + epsilon
    Q = Q + epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return divergence


# -------------------------------------------------------------------
# RANDOM HELPERS
# -------------------------------------------------------------------  

def plot_simple_fourier_decomposition():
    fig = plt.figure(figsize=(10, 5), constrained_layout=True);
    gs = fig.add_gridspec(6, 6);

    X = np.linspace(0.0, 1.0, 100)

    HZ_X = np.arange(1, 11, 1)
    HZ_Y = np.zeros(len(HZ_X))

    Y0 = np.sin(X * 2 * np.pi) * 0.5
    HZ_Y[0] = 0.5
    ax_Y0 = fig.add_subplot(gs[0:2, 0:2])
    sns.lineplot(ax=ax_Y0, x=X, y=Y0)
    ax_Y0.set_xticklabels([])
    ax_Y0.set_yticklabels([])

    Y1 = np.sin(3 * X * 2 * np.pi) * 0.35
    HZ_Y[2] = 0.35
    ax_Y1 = fig.add_subplot(gs[2:4, 0:2])
    sns.lineplot(ax=ax_Y1, x=X, y=Y1)
    ax_Y1.set_xticklabels([])
    ax_Y1.set_yticklabels([])

    Y2 = np.sin(5 * X * 2 * np.pi) * 0.15
    HZ_Y[4] = 0.15
    ax_Y2 = fig.add_subplot(gs[4:6, 0:2])
    sns.lineplot(ax=ax_Y2, x=X, y=Y2)
    ax_Y2.set_xlabel('Signal Components')
    ax_Y2.set_yticklabels([])
    ax_Y2.set_xticklabels([])

    Y = Y0 + Y1 + Y2
    ax_Y = fig.add_subplot(gs[0:3, 2:6])
    sns.lineplot(ax=ax_Y, x=X, y=Y)
    ax_Y.set_xlabel('Combined Signal')
    ax_Y.set_yticklabels([])
    ax_Y.set_xticklabels([])

    ax_HZ = fig.add_subplot(gs[3:6, 2:6])
    sns.barplot(ax=ax_HZ, x=HZ_X, y=HZ_Y)
    ax_HZ.set_xlabel('Frequency Spectrum')
    ax_HZ.set_yticklabels([])
    ax_HZ.set_xticklabels([])
    
    return fig
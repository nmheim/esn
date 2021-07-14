import matplotlib.pyplot as plt
from numpy import save, load, nan, arange, mean
import matplotlib.animation as animation

def animate_comparison(targets,predictions,kuro=True,filepath='comparison.mp4',fps=24,dpi=150, v=(None,None)):
    if v[0] is None:
        #min/max
        vmin = min(targets.min(),predictions.min())
        vmax = max(targets.max(),predictions.max())

    if kuro:
        mask=load('Kuroshio_mask.npy')
        if targets[0].shape!=mask.shape:
            print('Not Kuroshimo-shaped')

        else:
            targets[:,mask]=nan
            predictions[:,mask]=nan

    def init():
        im1.set_data(targets[0,:,:],)
        im2.set_data(predictions[0,:,:])

        return (im1,im2)

    # animation function. This is called sequentially
    def animate(i):
        target_slice = targets[i,:,:]
        prediction_slice = predictions[i,:,:]
        im1.set_data(target_slice)
        im2.set_data(prediction_slice)

        return (im1,im2)

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(8.5,4),dpi=dpi,sharey=True)
    fig.tight_layout()
    im1 = ax1.imshow(targets[0,:,:],cmap='inferno',origin='lower',vmin=vmin,vmax=vmax)
    im2 = ax2.imshow(predictions[0,:,:],cmap='inferno',origin='lower',vmin=vmin,vmax=vmax)
    #[left, bottom, width, height] 
    cbar_ax = fig.add_axes([0.92, 0.235, 0.03, 0.58])
    fig.colorbar(im1, cax=cbar_ax)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')
    ax1.grid(True,color='white')
    ax2.grid(True,color='white')
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    ax1.set_xlabel('Indices')
    ax2.set_xlabel('Indices')
    ax1.set_ylabel('Indices')
    
    #clear whitespace
    fig.subplots_adjust(
        left=0.09, 
        bottom=0.05, 
        right=0.9, 
        top=1, 
        wspace=0.05, 
        hspace=None)   
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=targets.shape[0], interval=20, blit=True)
    
    anim.save(
        filepath,
        writer=animation.FFMpegWriter(fps=fps),
        dpi=dpi,
        )
    
    return

def animate_this(anim_data, filepath='animation.mp4',fps=24,dpi=150):
    def init():
        im.set_data(anim_data[0,:,:])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = anim_data[i,:,:]
        im.set_data(data_slice)
        return (im,)
    fig, ax = plt.subplots()
    im = ax.imshow(anim_data[0,:,:],cmap='inferno',vmin=0,vmax=1)
    plt.colorbar(im)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=anim_data.shape[0], interval=20, blit=True)
    #clear whitespace
    fig.subplots_adjust(
        left=0, 
        bottom=0, 
        right=1, 
        top=1, 
        wspace=None, 
        hspace=None)
    anim.save(
        filepath,
        writer=animation.FFMpegWriter(fps=fps),
        dpi=dpi,
        )
    
    return anim


def MSE_over_time(targets,predictions,subplot_kw=None):
    SE = (targets-predictions)**2
    MSE_over_time = SE.mean(axis=1).mean(axis=1)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=subplot_kw,
        figsize=(5.5,2.74)
    )
    ax.plot(
        arange(1,len(targets)+1),
        MSE_over_time,
        color='black',
        linestyle='-',
        marker='.',
        markersize = 2,
        mec='red',
        mfc='red',
    )
    #standard labels if not set
    if len(ax.get_xlabel())<1:
        ax.set_xlabel('Time Step')
    if len(ax.get_ylabel())<1:
        ax.set_ylabel('MSE')
    
    return fig
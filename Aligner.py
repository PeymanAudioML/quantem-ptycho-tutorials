from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pyms

def choose_zone(d1,dimensions):
    
    fig, ax = plt.subplots(nrows=1)

    def plot(d1,fig,r1,r2):
        shape = d1.shape
        img1 = np.zeros(shape, dtype=np.uint8)

        ax = fig.axes
        ax[0].clear()
        ax[0].imshow(d1,cmap =plt.get_cmap('gray'))
        ax[0].set_title('Get specimen tilt:\nLine up the red dot with the zone-axis')
        ax[0].plot(*[x//2+y for x,y in zip(d1.shape,[r1,r2])][::-1],'ro')
        return fig


    class ButtonPressProcessor(object):

        def __call__(self, event):
            if event.key == 'right':
                self.r2 += 1
            elif event.key == 'left':
                self.r2 -= 1
            elif event.key == 'down':
                self.r1 += 1
            elif event.key == 'up':
                self.r1 -= 1
            plot(d1,self.fig,self.r1,self.r2)
            self.fig.canvas.draw()

        def __init__(self, fig):
            self.r1 = 0
            self.r2 = 0
            self.fig = fig
            self.fig.canvas.mpl_connect('key_press_event', self)

    fig = plot(d1,fig,0,0)
    B = ButtonPressProcessor(fig)
    plt.show(block=True)
    return np.asarray([B.r1/d1.shape[0]*dimensions[0],B.r2/d1.shape[1]*dimensions[1]])

def interactive_align(d1,d2,renorm=True,arcs=6):

    def renormalize(img):
        min_ = np.amin(img)
        max_ = np.amax(img)
        return (img - min_) / (max_ - min_)
    
    fig, ax = plt.subplots(nrows=2)
    r1 = 0
    r2 = 0


    def plot(d1,d2,fig,renorm=True):
        shape = d1.shape
        
        img1 = np.zeros(shape, dtype=np.uint8)

        if renorm:
            d1_ = renormalize(d1)*255
            d2_ = renormalize(d2)*255
        else:
            d1_ = d1
            d2_ = d2

        yy = np.broadcast_to(np.fft.fftfreq(d1.shape[0])[:, None], d1.shape[-2:])
        xx = np.broadcast_to(np.fft.fftfreq(d1.shape[1])[None, :], d1.shape[-2:])
        angle = np.fft.fftshift(np.arctan2(yy, xx) - np.pi / arcs / 3 * 2)
        mask = np.sin(angle / 2 * arcs) > 0
        ax = fig.axes
        img1[mask] = d1_[mask].astype(np.uint8)
        img1[np.logical_not(mask)] = d2_[np.logical_not(mask)].astype(np.uint8)
        ax[0].imshow(img1)
        ax[0].set_title('Get diffraction shift:\nAlign aperture function with PACBED')
        img2 = np.zeros(shape + (3,), dtype=np.uint8)
        img2[..., 0] = (renormalize(d1_)*255).astype(np.uint8)
        img2[..., 2] = (renormalize(d2_)*255).astype(np.uint8)
        ax[1].imshow(img2)
        for a in ax:
            a.set_axis_off()
        return fig


    class ButtonPressProcessor(object):

        def __call__(self, event):
            if event.key == 'right':
                self.r2 += 1
            elif event.key == 'left':
                self.r2 -= 1
            elif event.key == 'down':
                self.r1 += 1
            elif event.key == 'up':
                self.r1 -= 1
            plot(d1,np.roll(d2,(self.r1,self.r2),axis=(0,1)),self.fig)
            self.fig.canvas.draw()

        def __init__(self, fig):
            self.r1 = 0
            self.r2 = 0
            self.fig = fig
            self.fig.canvas.mpl_connect('key_press_event', self)

    fig = plot(d1,d2,fig)
    B = ButtonPressProcessor(fig)
    plt.show(block=True)
    return np.asarray([B.r1,B.r2])

if __name__ == "__main__":
    pix = [128,128]
    Smatrix_dimensions = [3.61, 3.61]
    eV = 3e5
    app = 20

    from pyms import focused_probe
    aperture_function = focused_probe(
        pix,
        np.asarray(pix) / np.asarray(Smatrix_dimensions),
        eV,
        app,
        qspace=True,
        tilt_units='pixels',
    )
    aperture_function = np.abs(np.fft.ifftshift(aperture_function))

    fnams = ['Inputs/PACBED.tif']
    for fnam in fnams:
        PACBED = np.asarray(Image.open(fnam))
        print(fnam)
        diff_shift = interactive_align(PACBED,aperture_function)
        print(diff_shift)
        mraddims = Smatrix_dimensions/pyms.wavev(eV)*1e3
        origin = np.asarray(diff_shift)/np.asarray(PACBED.shape)*mraddims
        print(fnam,choose_zone(PACBED,mraddims)-origin)
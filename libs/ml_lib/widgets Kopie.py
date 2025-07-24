
import ipywidgets as widgets
import numpy as np
from IPython.display import display

def model_gui(comp):
    coords = []
    cpairs = []
    
    vf_widget = widgets.Text(
        value='0.0 %', 
        description='Volume fraction filler:',
        disabled=True, 
        style={'description_width': '160px'}
    )

    shape_widget = widgets.Text(
        value='X0: 0.0, X1: 0.0 | Y0: 0.0, Y1: 0.0',
        description='Shape parameters:',
        disabled=True, 
        style={'description_width': '160px'},
        layout=widgets.Layout(width='500px')
    )

    update_btn = widgets.Button(description="Update", layout=widgets.Layout(width='210px'))
    erase_btn = widgets.Button(description="Erase", layout=widgets.Layout(width='210px'))
    finish_btn = widgets.Button(description="Finish", layout=widgets.Layout(width='210px'))

    def update_info_widget(vf, shp):
        vf_widget.value = f'{vf*100:5.1f}%'
        shape_widget.value = (
            f'X0: {shp[0, 0]:5.3f}, X1: {shp[0, 1]:5.3f} | '
            f'Y0: {shp[1, 0]:5.3f}, Y1: {shp[1, 1]:5.3f}'
        )

    def update_model(button):
        nonlocal coords, cpairs, fig, ax
        x0, y0, x1, y1 = comp.upd_model(coords)
        cpairs.append([x0, y0, x1, y1])
        fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
        vf, shape = comp.calc_geom_param()
        update_info_widget(vf, shape)
        coords = []

    def erase_model(button):
        nonlocal coords, cpairs, fig, ax
        comp.create_model()
        fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
        coords = []
        cpairs = []
        vf = 0.0
        shape = np.zeros((2, 2), dtype=int)
        update_info_widget(vf, shape)

    def finish_model(button):
        nonlocal coords, cpairs, fig, ax, cid
        if len(coords) > 1:
            x0, y0, x1, y1 = comp.upd_model(coords)
            cpairs.append([x0, y0, x1, y1])
            fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
            vf, shape = comp.calc_geom_param()
            update_info_widget(vf, shape)
        coords = []
        fig.canvas.mpl_disconnect(cid)
        update_btn.disabled = True
        erase_btn.disabled = True
        finish_btn.disabled = True

    def coord_click(event):
        nonlocal coords
        coords.append([event.xdata, event.ydata])

    update_btn.on_click(update_model)
    erase_btn.on_click(erase_model)
    finish_btn.on_click(finish_model)

    ui_comp = widgets.VBox([
        widgets.HBox([vf_widget, shape_widget]),
        widgets.HBox([update_btn, erase_btn, finish_btn])
    ], layout=widgets.Layout(padding='10px', margin='20px'))

    fig, ax = comp.plot('mat', mag=1)
    update_info_widget(vf=0.0, shp=np.zeros((2, 2), dtype=int))

    cid = fig.canvas.mpl_connect('button_press_event', coord_click)
    update_btn.disabled = False
    erase_btn.disabled = False
    finish_btn.disabled = False

    return ui_comp, fig, ax, coords, cpairs, cid
    


""" Define global functions for widget actions """

# Labels for boxes in user interface for parameter definitions
label_model = widgets.Label(value="Model", 
                            style = {'background' : '#CEDF9F','font_size':'19px','font_weight':'bold'},
                            layout=widgets.Layout(padding = '0px 80px'))
label_material = widgets.Label(value="Material properties", 
                               style = {'background' : '#CEDF9F','font_size':'19px','font_weight':'bold'},
                               layout=widgets.Layout(padding = '0px 80px'))
label_mesh = widgets.Label(value="Mesh", 
                           style = {'background' : '#EAF5B8','font_size':'17px'},
                           layout=widgets.Layout(padding = '0px 80px'))
label_bc = widgets.Label(value="Boundary Conditions", 
                         style = {'background' : '#EAF5B8','font_size':'17px'},
                         layout=widgets.Layout(padding = '0px 80px'))
label_matrix = widgets.Label(value="Matrix", 
                             style = {'background' : '#EAF5B8','font_size':'17px'},
                             layout=widgets.Layout(padding = '0px 80px'))
label_filler = widgets.Label(value="Filler", 
                             style = {'background' : '#EAF5B8','font_size':'17px'},
                             layout=widgets.Layout(padding = '0px 80px'))

# Widgets for model and material parameter definitions
nel_widget = widgets.IntText(value=30, description="Elements per side: ", style={'description_width':'160px'})
sides_widget = widgets.Dropdown(description="Lateral boundary: ",options=['free', 'fixed'], style={'description_width':'160px'})
etot_widget = widgets.FloatText(value=1, description="Total strain (%): ", style={'description_width':'160px'})

E1_widget = widgets.FloatText(value=10.0, description="Young's modulus (GPa): ", style={'description_width':'160px'})
nu1_widget = widgets.FloatText(value=0.27, description="Poisson's ratio (.): ", style={'description_width':'160px'})
E2_widget = widgets.FloatText(value=300.0, description="Young's modulus (GPa): ", style={'description_width':'160px'})
nu2_widget = widgets.FloatText(value=0.27, description="Poisson's ratio (.): ", style={'description_width':'160px'})

model_box = widgets.VBox([label_model, 
                          label_mesh, nel_widget,
                          label_bc, etot_widget, sides_widget], 
                         layout=widgets.Layout(border='solid 2px #A1D6B2', padding='10px', margin='10px'))
material_box = widgets.VBox([label_material, 
                             label_matrix, E1_widget, nu1_widget,
                             label_filler, E2_widget, nu2_widget], 
                            layout=widgets.Layout(border='solid 2px #A1D6B2', padding='10px', margin='10px'))

#define user interfaces
ui_mat = widgets.VBox([widgets.HBox([material_box, model_box])], layout=widgets.Layout(padding='10px', margin='20px'))

                   
def model_gui(comp):
    coords = []
    cpairs = []
    def update_info_widget(vf, shp):
        # global vf_widget, shape_widget
        vf_widget.value = f'{vf*100:5.1f}%'
        shape_widget.value = f'X0: {shp[0, 0]:5.3f}, X1: {shp[0, 1]:5.3f} | Y0: {shp[1, 0]:5.3f}, Y1: {shp[1, 1]:5.3f}'


    def update_model(button):
        # global comp, coords, cpairs, fig, ax
        x0, y0, x1, y1 = comp.upd_model(coords)
        cpairs.append([x0, y0, x1, y1])
        fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
        vf, shape = comp.calc_geom_param()
        update_info_widget(vf, shape)
        coords = []  # reset clicked coordinates
    
        
    def erase_model(button):
        # global comp, coords, cpairs, fig, ax
        comp.create_model()
        fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
        coords = []
        cpairs = []
        vf = 0.0
        shape = np.zeros((2,2), dtype=int)
        update_info_widget(vf, shape)
    
    
    def finish_model(button):
        # global comp, coords, cpairs, fig, ax, cid
        if len(coords) > 1:
            x0, y0, x1, y1 = comp.upd_model(coords)
            cpairs.append([x0, y0, x1, y1])
            fig, ax = comp.plot(val='mat', mag=1, fig=fig, ax=ax)
            vf, shape = comp.calc_geom_param()
            update_info_widget(vf, shape)
        coords = []
        #deactivate buttons for user interaction
        fig.canvas.mpl_disconnect(cid)
        update_btn.disabled = True
        erase_btn.disabled = True
        finish_btn.disabled = True
    
    
    def coord_click(event):
        # global coords
        coords.append([event.xdata, event.ydata])
    
    
    """  Define variables and boxes for widgets  """
    
    # Info widgets for results on geometrical arrangement of filler phase
    vf_widget = widgets.Text(value='0.0 %', 
                             description='Volume fraction filler:',
                             disabled=True, style={'description_width':'160px'})
    shape_widget = widgets.Text(value='X0: 0.0, X1: 0.0 | Y0: 0.0, Y1: 0.0',
                                description='Shape parameters:',
                                disabled=True, style={'description_width':'160px'},
                                layout=widgets.Layout(width='500px'))
    
    # Buttons for interactive material definition
    update_btn = widgets.Button(description="Update", disabled=False, layout=widgets.Layout(width='210px'))
    erase_btn = widgets.Button(description="Erase", disabled=False, layout=widgets.Layout(width='210px'))
    finish_btn = widgets.Button(description="Finish", disabled=False, layout=widgets.Layout(width='210px'))
    
    # Link buttons to their respective functions
    update_btn.on_click(update_model)
    erase_btn.on_click(erase_model)
    finish_btn.on_click(finish_model)
    
    ui_comp = widgets.VBox([widgets.HBox([vf_widget, shape_widget]), 
                        widgets.HBox([update_btn, erase_btn, finish_btn])],
                       layout=widgets.Layout(padding='10px', margin='20px'))
    fig, ax = comp.plot('mat', mag=1)  # plot model geometry
    update_info_widget(vf=0.0, shp=np.zeros((2,2), dtype=int))
    
    # activate buttons for user interaction
    cid = fig.canvas.mpl_connect('button_press_event', coord_click)  # record user clicks in model
    update_btn.disabled = False
    erase_btn.disabled = False
    finish_btn.disabled = False
    
    return ui_comp, fig, ax, coords, cpairs, cid
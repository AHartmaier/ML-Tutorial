import ipywidgets as widgets
import numpy as np
from IPython.display import display

def model_gui(comp):
    # shared state variable
    state = {
        "coords": [],
        "cpairs": [],
        "fig": None,
        "ax": None,
        "cid": None
    }

    # Info widgets
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

    # Buttons
    update_btn = widgets.Button(description="Update", layout=widgets.Layout(width='210px'))
    erase_btn = widgets.Button(description="Erase", layout=widgets.Layout(width='210px'))
    finish_btn = widgets.Button(description="Finish", layout=widgets.Layout(width='210px'))

    # Funktionen
    def update_info_widget(vf, shp):
        vf_widget.value = f'{vf*100:5.1f}%'
        shape_widget.value = (
            f'X0: {shp[0, 0]:5.3f}, X1: {shp[0, 1]:5.3f} | '
            f'Y0: {shp[1, 0]:5.3f}, Y1: {shp[1, 1]:5.3f}'
        )

    def update_model(button):
        if not state["coords"]:
            return
        x0, y0, x1, y1 = comp.upd_model(state["coords"])
        state["cpairs"].append([x0, y0, x1, y1])
        state["fig"], state["ax"] = comp.plot(val='mat', mag=1, fig=state["fig"], ax=state["ax"])
        vf, shape = comp.calc_geom_param()
        update_info_widget(vf, shape)
        state["coords"] = []

    def erase_model(button):
        comp.create_model()
        state["fig"], state["ax"] = comp.plot(val='mat', mag=1, fig=state["fig"], ax=state["ax"])
        state["coords"] = []
        state["cpairs"] = []
        update_info_widget(0.0, np.zeros((2, 2), dtype=int))

    def finish_model(button):
        if len(state["coords"]) > 1:
            x0, y0, x1, y1 = comp.upd_model(state["coords"])
            state["cpairs"].append([x0, y0, x1, y1])
            state["fig"], state["ax"] = comp.plot(val='mat', mag=1, fig=state["fig"], ax=state["ax"])
            vf, shape = comp.calc_geom_param()
            update_info_widget(vf, shape)
        state["coords"] = []
        state["fig"].canvas.mpl_disconnect(state["cid"])
        update_btn.disabled = True
        erase_btn.disabled = True
        finish_btn.disabled = True

    def coord_click(event):
        if event.xdata is not None and event.ydata is not None:
            state["coords"].append([event.xdata, event.ydata])

    # Button-Callbacks verbinden
    update_btn.on_click(update_model)
    erase_btn.on_click(erase_model)
    finish_btn.on_click(finish_model)

    # GUI zusammensetzen
    ui_comp = widgets.VBox([
        widgets.HBox([vf_widget, shape_widget]),
        widgets.HBox([update_btn, erase_btn, finish_btn])
    ], layout=widgets.Layout(padding='10px', margin='20px'))

    # Initialize
    state["fig"], state["ax"] = comp.plot('mat', mag=1)
    update_info_widget(0.0, np.zeros((2, 2), dtype=int))
    state["cid"] = state["fig"].canvas.mpl_connect('button_press_event', coord_click)

    return ui_comp, state["fig"], state["ax"], state["coords"], state["cpairs"], state["cid"]
    

class mat_gui(object):
    def __init__(self):
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
        self.nel_widget = widgets.IntText(value=30, description="Elements per side: ", style={'description_width':'160px'})
        self.sides_widget = widgets.Dropdown(description="Lateral boundary: ",options=['free', 'fixed'], style={'description_width':'160px'})
        self.etot_widget = widgets.FloatText(value=1, description="Total strain (%): ", style={'description_width':'160px'})
        
        self.E1_widget = widgets.FloatText(value=10.0, description="Young's modulus (GPa): ", style={'description_width':'160px'})
        self.nu1_widget = widgets.FloatText(value=0.27, description="Poisson's ratio (.): ", style={'description_width':'160px'})
        self.E2_widget = widgets.FloatText(value=300.0, description="Young's modulus (GPa): ", style={'description_width':'160px'})
        self.nu2_widget = widgets.FloatText(value=0.27, description="Poisson's ratio (.): ", style={'description_width':'160px'})
        
        model_box = widgets.VBox([label_model, 
                                  label_mesh, self.nel_widget,
                                  label_bc, self.etot_widget, self.sides_widget], 
                                 layout=widgets.Layout(border='solid 2px #A1D6B2', padding='10px', margin='10px'))
        material_box = widgets.VBox([label_material, 
                                     label_matrix, self.E1_widget, self.nu1_widget,
                                     label_filler, self.E2_widget, self.nu2_widget], 
                                    layout=widgets.Layout(border='solid 2px #A1D6B2', padding='10px', margin='10px'))
        
        #define user interfaces
        self.table = widgets.VBox([widgets.HBox([material_box, model_box])], layout=widgets.Layout(padding='10px', margin='20px'))

        
    def read_param(self):
        param = {"nel": self.nel_widget.value,
                 "etot": self.etot_widget.value,
                 "latbc": self.sides_widget.value,
                 "E1": self.E1_widget.value,
                 "nu1": self.nu1_widget.value,
                 "E2": self.E2_widget.value,
                 "nu2": self.nu2_widget.value}
        return param

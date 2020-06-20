# Author: Taha Nakabi
from tcl_env_dqn_1 import MicroGridEnv
import pygal as pg
import numpy as np
from pygal.style import Style
custom_style = Style(legend_font_size=10)
RENDER_VALUES_list=['Energy Generated', 'Energy consumption' ,
                    'TCL SOCs', 'Outdoor temperatures (°C)' ,
                    'Price responsive Loads (kW)','Retail Sale prices (cents)',
                    'Battery SOC',
                    'Energy Sold','Energy purchased',
                    'Up regulation prices','Down-regulation Prices',
                    'Energy allocated to TCLs','Energy consumed by TCLs']
RENDER_VALUES_dict={}
def reset_dict():
    for key in RENDER_VALUES_list:
        RENDER_VALUES_dict[key]=[]

reset_dict()


class MicroGridEnvWeb(MicroGridEnv):

    def __init__(self,**kwargs):
        MicroGridEnv.__init__(self, **kwargs)

    def add_content(self, ax,i):
        ax.render_to_file('svgs/graph'+str(i)+'.svg')
        with open('svgs/graph'+str(i)+'.svg', "r") as f:
            rows = f.readlines()[1:]
        for row in rows:
            self.html_file.write(row)

    def render(self,name=''):

        values=[self.generation.current_generation(self.day*self.iterations+self.time_step-1),
                self._compute_tcl_power() + np.sum([l.load() for l in self.loads]),
                np.average([tcl.SoC for tcl in self.tcls])*100,
                self.temperatures[self.day*self.iterations+self.time_step-1],
                np.average([l.load() for l in self.loads]),
                self.sale_price,
                self.battery.SoC*100,
                self.energy_sold,
                self.energy_bought,
                self.grid.buy_prices[self.day * self.iterations + self.time_step-1],
                self.grid.sell_prices[self.day * self.iterations + self.time_step-1],
                self.control,
                self._compute_tcl_power()]
        for index, key in enumerate(RENDER_VALUES_dict.keys()):
            RENDER_VALUES_dict[key].append(values[index])

        if self.time_step==self.iterations:
            self.html_file = open("app/templates/figure.html", 'w')
            self.html_file.write("{% extends \"base.html\" %}" + "\n"+ "{% block content %}" + "\n")
            self.html_file.write("<h1>Day: {}</h1>".format(self.day))

            ax = pg.Line(height=300, include_x_axis=True, label_font_size=2,truncate_legend=-1,
                        title_font_size=20, x_title="Time steps",
                        y_title="Energy"+" (kWh)",
                        legend_at_bottom=True, x_label_rotation=0)
            ax.title =  "Energy generated and consumed"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[0],RENDER_VALUES_dict[RENDER_VALUES_list[0]])
            ax.add(RENDER_VALUES_list[1], RENDER_VALUES_dict[RENDER_VALUES_list[1]])
            self.add_content(ax,1)

            ax = pg.Line(height=300,include_x_axis=True, label_font_size=1,truncate_legend=-1,
                        title_font_size=20, x_title="Time steps",
                        y_title="TCLs SOC",
                        legend_at_bottom=False, x_label_rotation=0)
            ax.title = "Average TCLs state of charge % and outdoor temperatures"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[2], RENDER_VALUES_dict[RENDER_VALUES_list[2]])
            ax.add(RENDER_VALUES_list[3], RENDER_VALUES_dict[RENDER_VALUES_list[3]], secondary=True)
            self.add_content(ax,2)

            ax = pg.Line(height=300, include_x_axis=True, label_font_size=1,truncate_legend=-1,
                        title_font_size=20, x_title="Time steps", spacing=1,
                        legend_at_bottom=False, x_label_rotation=0, style=custom_style)
            ax.title = "Retail prices and average consumption from price responsive loads"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[5], RENDER_VALUES_dict[RENDER_VALUES_list[5]])
            ax.add(RENDER_VALUES_list[4], RENDER_VALUES_dict[RENDER_VALUES_list[4]], secondary=True)
            self.add_content(ax,3)

            ax = pg.Line(height=300, include_x_axis=True, label_font_size=1,truncate_legend=-1,
                         title_font_size=20, x_title="Time steps",
                         show_legend=False, x_label_rotation=0, range=(0,100))
            ax.title = "State of charge of the energy storage system  %"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[6], RENDER_VALUES_dict[RENDER_VALUES_list[6]])
            self.add_content(ax, 4)

            ax = pg.Bar(height=300, include_x_axis=True, label_font_size=2,truncate_legend=-1,
                         title_font_size=20, x_title="Time steps",
                         legend_at_bottom=True, x_label_rotation=0)
            ax.title = "Energy sold to and purchased from the main grid (kWh)"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[8], RENDER_VALUES_dict[RENDER_VALUES_list[8]])
            ax.add(RENDER_VALUES_list[7], RENDER_VALUES_dict[RENDER_VALUES_list[7]])
            self.add_content(ax, 5)

            ax = pg.Line(height=300, include_x_axis=True, label_font_size=2,truncate_legend=-1,
                        title_font_size=20, x_title="Time steps",
                        legend_at_bottom=True, x_label_rotation=0)
            ax.title = "Electricity prices in the balancing market (cents/kW)"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[9], RENDER_VALUES_dict[RENDER_VALUES_list[9]])
            ax.add(RENDER_VALUES_list[10], RENDER_VALUES_dict[RENDER_VALUES_list[10]])
            self.add_content(ax, 6)

            ax = pg.Bar(height=300, include_x_axis=True, label_font_size=2 , truncate_legend=-1,
                        title_font_size=20, x_title="Time steps",
                        legend_at_bottom=True, x_label_rotation=0)
            ax.title = "Energy allocated to and consumed by TCLs (kWh)"
            ax.x_labels = range(self.iterations)
            ax.add(RENDER_VALUES_list[12], RENDER_VALUES_dict[RENDER_VALUES_list[12]])
            ax.add(RENDER_VALUES_list[11], RENDER_VALUES_dict[RENDER_VALUES_list[11]])
            self.add_content(ax, 7)



            self.html_file.write("\n <form action=\"\" method=\"post\" novalidate> \n"+
                                 "{{ form.hidden_tag() }} \n " +
                                 "<p>{{ form.next_day() }}</p>\n " )
            if self.day!=self.day0:
                self.html_file.write("<p>{{ form.previous_day() }}</p>\n ")

            self.html_file.write(" </form>")
            self.html_file.write("{% endblock %}")
            self.html_file.close()
            reset_dict()


        ## TO DO: write a render method using pygal
        pass
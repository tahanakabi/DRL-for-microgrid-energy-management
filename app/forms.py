from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, IntegerField
from wtforms.validators import Optional



class ParamForm(FlaskForm):
    # parameters_list = ["iterations",
    #                    "num_tcls"
    #                    "tcl_power",
    #                    "tcl_price",
    #                    "num_loads",
    #                    "base_load",
    #                    "normal_price",
    #                    "temperatures",
    #                    "price_tiers",
    #                    "day0",
    #                    "dayn",
    #                    "power_cost",
    #                    "down_reg",
    #                    "up_reg",
    #                    "imp_fees",
    #                    "exp_fees",
    #                    "battery_capacity",
    #                    "max_discharge",
    #                    "max_charge"]
    iterations = IntegerField("Number of timesteps",validators=[Optional()])
    num_tcls = IntegerField("Number of TCLs",validators=[Optional()])
    tcl_power = FloatField("TCL average power kW",validators=[Optional()])
    tcl_price = FloatField("TCLs price",validators=[Optional()])
    num_loads = IntegerField("Number of loads",validators=[Optional()])
    # base_load = FloatField("Typical load",validators=[Optional()])
    normal_price = FloatField("Normal retail price",validators=[Optional()])
    # temperatures = FloatField("Temperatures",validators=[Optional()])
    day0 = IntegerField("Day 0",validators=[Optional()])
    # dayn = IntegerField("Day N",validators=[Optional()])
    power_cost = FloatField("power cost per kWh",validators=[Optional()])
    # down_reg = FloatField("Down regulation prices",validators=[Optional()])
    # up_reg = FloatField("Up regulation prices",validators=[Optional()])
    imp_fees = FloatField("Power Transmission fees (import) ",validators=[Optional()])
    exp_fees = FloatField("Power Transmission fees (export) ",validators=[Optional()])
    battery_capacity = FloatField("Battery Capacity kWh",validators=[Optional()])
    max_discharge = FloatField("Discharge limit per hour",validators=[Optional()])
    max_charge = FloatField("Charge limit per hour",validators=[Optional()])
    submit = SubmitField('Submit')
    # parameters_dict={}
    # for param in parameters_list:
    #     parameters_dict[param]= FloatField(param)

class NextDayForm(FlaskForm):
    next_day = SubmitField("Next Day")
    previous_day = SubmitField("Previous Day")
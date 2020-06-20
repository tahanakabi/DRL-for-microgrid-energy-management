from app import app
from app.forms import ParamForm, NextDayForm
from flask import render_template, flash, redirect, url_for, session
from flask import request
from A3C_plusplus import Environment, Brain, MODELS_DIRECTORY
from Visualize import line_chart
from microgrid_env_web import MicroGridEnvWeb
import numpy as np
import pickle


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title= 'Home')


# @app.route('/parameters')
# def parameters():
#     form = ParamForm()
#     return render_template('parameters.html', title='Microgrid Parameters', form=form)

@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    form = ParamForm(request.form)
    session.pop('enviro', None)
    # param_dict={}
    if request.method == "POST" and form.validate_on_submit():
        flash('The following data is being processed: \n')
        for fieldname, value in form.data.items():
            if fieldname!= "submit" and fieldname!="csrf_token" and value is not None:
                flash('{}={}'.format(fieldname, value))
                session[fieldname]=value
        flash('The rest of the parameters are going to take the default values')
        return redirect(url_for('graphs'))
    else:
        return render_template('parameters.html', title='Microgrid Parameters', form=form)

@app.route('/graphs',  methods=['GET', 'POST'])
def graphs():
    form = NextDayForm(request.form)
    if form.validate_on_submit() and request.method == "POST" and form.next_day.data:
        return redirect(url_for('next_graphs'))
    print("Initializing the environment")
    global enviro
    enviro= Environment(render=True, eps_start=0., eps_end=0.)
    print("Initializing the environment's environmennt")
    enviro.env = MicroGridEnvWeb(**session)
    print("Initializing the Brain")
    brain = Brain(environment=enviro)
    print("Associating the brain with the environment")
    enviro.brain=brain
    print("Running the episode")
    enviro.runEpisode()
    print("Rendering the template")
    return render_template('figure.html', title="Results", form= form)

@app.route('/next_graphs',  methods=['GET', 'POST'])
def next_graphs():
    print('next graphs')
    form = NextDayForm(request.form)
    print(form.next_day)
    if form.validate_on_submit() and request.method == "POST" and form.next_day.data:
        return redirect(url_for('next_graphs'))
    if form.validate_on_submit() and request.method == "POST" and form.previous_day.data:
        return redirect(url_for('previous_graphs'))
    day = enviro.env.day + 1
    enviro.runEpisode(day)
    print(enviro.env.day)
    return render_template('figure.html', title="Results", form=form)

@app.route('/previous_graphs',  methods=['GET', 'POST'])
def previous_graphs():
    print('previous graphs')
    form = NextDayForm(request.form)

    if form.validate_on_submit() and request.method == "POST" and form.previous_day.data:
        return redirect(url_for('previous_graphs'))
    if form.validate_on_submit() and request.method == "POST" and form.next_day.data:
        return redirect(url_for('next_graphs'))
    day = enviro.env.day - 1
    enviro.env.reset_all(day)
    enviro.runEpisode(day)
    print(enviro.env.day)
    return render_template('figure.html', title="Results", form=form)

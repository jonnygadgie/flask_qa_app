import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import scipy.optimize
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


# pretty much all the code I have is from https://morioh.com/p/20750b8a8580
# Now, (06/02/2021) I will modify the code to run growth curve modelling

def get_db_connection():
    conn = sqlite3.connect('database3.db')
    conn.row_factory = sqlite3.Row
    return conn   # connects to database and allows you to call column names
    # database now returns rows that act like normal Python dictionaries


def get_post(post_id):                        # Determine which post to display on single page
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?',
                        (post_id,)).fetchone()
    # Now, (06/10/2021) I will modify the code to run growth curve modelling
    # Pull in data from the database - this has already been done three lines above

    rex = post # 'post' is the entire row of data in the database. Each column has specific user inputted data
    Pre_years = rex[3]                  # string of years input by user
    Pre_datah = rex[4]                  # string of datapoints input by user
    Pre_parameters = rex[5]             # string of parameter guesses input by user
 
    first_years = list(map(int,Pre_years.split(",")))           # convert user strings to usable datatypes (int, float)
    first_datah = list(map(float,Pre_datah.split(",")))
    first_parameters = list(map(float,Pre_parameters.split(",")))
    # The last few lines were just getting the database values in a format I can use

    # Separate the columns of data into their own arrays. These are panda arrays and will need to be changed to numpy arrays (allows me to do matrix operations)
    # to do this, pull the second column from each dataframe and call the variables with '.values' when used in a function
    years = pd.DataFrame(data=first_years)
    datah = pd.DataFrame(data=first_datah)
    parameters = pd.DataFrame(data=first_parameters)
    # If I were to print these variables, they would be vertical columns of data
    # Now, (06/10/2021) I will modify the code to run growth curve modelling
    # Pull in data from the database

    #conn = sqlite3.connect('database3.db')
    #c = conn.cursor()
    #c.execute('SELECT * FROM posts WHERE id = 3')
    # rex = c.fetchone()
    # rex = post # post is defined by get_post(id). It's the row in the database specified by website user
    Pre_years = rex[3]
    Pre_datah = rex[4]
    Pre_parameters = rex[5]
    print(rex)
    print(Pre_parameters)
    first_years = list(map(int,Pre_years.split(",")))
    first_datah = list(map(float,Pre_datah.split(",")))
    first_parameters = list(map(float,Pre_parameters.split(",")))
    # The last few lines were just getting the database values in a format I can use

    # Separate the columns of data into their own arrays. These are panda arrays and will need to be changed to numpy arrays
    # to do this, pull the second column from each dataframe and call the variables with '.values' when used in a function
    years = pd.DataFrame(data=first_years)
    datah = pd.DataFrame(data=first_datah)
    parameters = pd.DataFrame(data=first_parameters)
    numcurves = first_parameters[0]
    x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user 


    if(numcurves < 3):
        # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array
        def ackley(x):
            arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
            arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
            arg3 = np.add(arg1, arg2)                      # add curves together to get total model curve
            arg4 = np.subtract(datah.values,arg3)          # find difference of this array and the actual data
            arg5 = np.square(arg4)                         # Square the difference
            return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)

        res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
        res.x

        # Define variables used in logistic model - res.x is the array of parameters (This example has 2 curves)
        L1 = res.x[0]
        H1 = res.x[1]
        W1 = res.x[2]
        M1 = res.x[3]
        L2 = res.x[4]
        H2 = res.x[5]
        W2 = res.x[6]
        M2 = res.x[7]

        # Use the Growth Curve Model to create curve1, curve2, sumcurves arrays
        curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
        curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
        sumcurves=curve1+curve2
        Li1 = str(L1)
        Hi1 = str(H1)
        Wi1 = str(W1)
        Mi1 = str(M1)
        Li2 = str(L2)
        Hi2 = str(H2)
        Wi2 = str(W2)
        Mi2 = str(M2)

        finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2])
        print(finalparametri)
        conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])
        # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
        # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
        allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
        alldatah = datah[0]                                
        plt.title("Growth Curve Modelling") 
        plt.xlabel("Years") 
        plt.ylabel("Efficiency") 
        plt.plot(allyears.values,curve1)
        plt.plot(allyears.values,curve2) 
        plt.plot(allyears.values,sumcurves)
        plt.plot(allyears.values,alldatah.values) 
        plt.legend(['Curve1', 'Curve2', 'SumCurves', 'Data'])  
        plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
        plt.show()
        # Commit and close database - actually comment this out as we do it later
        conn.commit()
        conn.close()




    if(numcurves == 3):
    # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array
        def ackley(x):
            arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
            arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
            arg3 = x[9]+((x[8]-x[9])*np.reciprocal(1+np.exp(((years.values)-x[11])/x[10])))  # Curve 3 growth curve model
            arg4b = np.add(arg1, arg2)
            arg4 = np.add(arg4b,arg3)                      # add curves together to get total model curve
            arg5 = np.subtract(datah.values,arg4)          # find difference of this array and the actual data
            arg6 = np.square(arg5)                         # Square the difference
            return np.sum(arg6)                            # add up all elements in the squared difference (sum of squares)

        x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user 
        res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
        res.x

        # Define variables used in logistic model - res.x is the array of parameters (This example has 2 curves)
        L1 = res.x[0]
        H1 = res.x[1]
        W1 = res.x[2]
        M1 = res.x[3]
        L2 = res.x[4]
        H2 = res.x[5]
        W2 = res.x[6]
        M2 = res.x[7]
        L3 = res.x[8]
        H3 = res.x[9]
        W3 = res.x[10]
        M3 = res.x[11]

        # Use the Growth Curve Model to create curve1, curve2, sumcurves arrays
        curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
        curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
        curve3 = H3+((L3-H3)*np.reciprocal(1+np.exp((years.values-M3)/W3)))
        sumcurvesb=curve1+curve2
        sumcurves = sumcurvesb+curve3
        Li1 = str(L1)
        Hi1 = str(H1)
        Wi1 = str(W1)
        Mi1 = str(M1)
        Li2 = str(L2)
        Hi2 = str(H2)
        Wi2 = str(W2)
        Mi2 = str(M2)
        Li3 = str(L3)
        Hi3 = str(H3)
        Wi3 = str(W3)
        Mi3 = str(M3)

        finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2, Li3, Hi3, Wi3, Mi3])

        print(finalparametri)
        conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])

        # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
        # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
        allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
        alldatah = datah[0]                                
        plt.title("Growth Curve Modelling") 
        plt.xlabel("Years") 
        plt.ylabel("Efficiency") 
        plt.plot(allyears.values,curve1)
        plt.plot(allyears.values,curve2) 
        plt.plot(allyears.values,curve3) 
        plt.plot(allyears.values,sumcurves)
        plt.plot(allyears.values,alldatah.values) 
        plt.legend(['Curve1', 'Curve2', 'Curve3', 'SumCurves', 'Data'])  
        plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
        plt.show()
        # Commit and close database - actually comment this out as we do it later
        conn.commit()
        conn.close()






    if(numcurves == 4):
        # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array
        def ackley(x):
            arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
            arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
            arg3 = x[9]+((x[8]-x[9])*np.reciprocal(1+np.exp(((years.values)-x[11])/x[10])))  # Curve 3 growth curve model
            arg3b = x[13]+((x[12]-x[13])*np.reciprocal(1+np.exp(((years.values)-x[15])/x[14])))  # Curve 3 growth curve model
            arg3c = np.add(arg1, arg2)                      # add curves together to get total model curve
            arg3d = np.add(arg3,arg3b)
            arg3e = np.add(arg3c,arg3d)
            arg4 = np.subtract(datah.values,arg3e)          # find difference of this array and the actual data
            arg5 = np.square(arg4)                         # Square the difference
            return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)

        res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
        res.x

        # Define variables used in logistic model - res.x is the array of parameters (This example has 2 curves)
        L1 = res.x[0]
        H1 = res.x[1]
        W1 = res.x[2]
        M1 = res.x[3]
        L2 = res.x[4]
        H2 = res.x[5]
        W2 = res.x[6]
        M2 = res.x[7]
        L3 = res.x[8]
        H3 = res.x[9]
        W3 = res.x[10]
        M3 = res.x[11]
        L4 = res.x[12]
        H4 = res.x[13]
        W4 = res.x[14]
        M4 = res.x[15]

        # Use the Growth Curve Model to create curve1, curve2, sumcurves arrays
        curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
        curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
        curve3 = H3+((L3-H3)*np.reciprocal(1+np.exp((years.values-M3)/W3)))
        curve4 = H4+((L4-H4)*np.reciprocal(1+np.exp((years.values-M4)/W4)))

        sumcurvesa=curve1+curve2
        sumcurvesb=curve3+curve4
        sumcurves=sumcurvesa+sumcurvesb
        Li1 = str(L1)
        Hi1 = str(H1)
        Wi1 = str(W1)
        Mi1 = str(M1)
        Li2 = str(L2)
        Hi2 = str(H2)
        Wi2 = str(W2)
        Mi2 = str(M2)
        Li3 = str(L3)
        Hi3 = str(H3)
        Wi3 = str(W3)
        Mi3 = str(M3)
        Li4 = str(L4)
        Hi4 = str(H4)
        Wi4 = str(W4)
        Mi4 = str(M4)

        finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2, Li3, Hi3, Wi3, Mi3, Li4, Hi4, Wi4, Mi4])
        print(finalparametri)
        conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])
        # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
        # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
        allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
        alldatah = datah[0]                                
        plt.title("Growth Curve Modelling") 
        plt.xlabel("Years") 
        plt.ylabel("Efficiency") 
        plt.plot(allyears.values,curve1)
        plt.plot(allyears.values,curve2) 
        plt.plot(allyears.values,curve3) 
        plt.plot(allyears.values,curve4) 
        plt.plot(allyears.values,sumcurves)
        plt.plot(allyears.values,alldatah.values) 
        plt.legend(['Curve1', 'Curve2', 'Curve3', 'Curve4', 'SumCurves', 'Data'])  
        plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
        plt.show()
        # Commit and close database - actually comment this out as we do it later
        conn.commit()
        conn.close()

    # Commit and close database 
    #conn.commit()
    #conn.close() 
    if post is None:
        abort(404)              # If you ask for a post which does not exist (no I.D.) - 404 error
    return post



app = Flask(__name__)
app.config['SECRET_KEY'] = 'thesecretestykey'


@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)
    # fetcha all rows of query result. Return result of rendering index.html template


@app.route('/<int:post_id>')
def post(post_id):
    post = get_post(post_id)
    return render_template('post.html', post=post)


@app.route('/create', methods=('GET', 'POST'))         # send to HTML template - what user sees when creating new post
def create():
    if request.method == 'POST':
        title = request.form['title']
        years = request.form['years']
        datas = request.form['datas']                  # setting up forms for the user to input data and stuff
        parameters = request.form['parameters']
        finalparameters = request.form['finalparameters']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO posts (title, years, datas, parameters, finalparameters) VALUES (?, ?, ?, ?, ?)',
                         (title, years, datas, parameters, finalparameters))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('create.html')


@app.route('/<int:id>/edit', methods=('GET', 'POST'))   # Edit an existing post
def edit(id):
    post = get_post(id)                                 # Get the post

    if request.method == 'POST':
        title = request.form['title']
        years = request.form['years']
        datas = request.form['datas']
        parameters = request.form['parameters']              # define title, years, etc.
        finalparameters = request.form['finalparameters']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE posts SET title = ?, years = ?, datas = ?, parameters = ?, finalparameters = ?'
                         ' WHERE id = ?',
                         (title, years, datas, parameters, finalparameters, id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('edit.html', post=post)


@app.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post['title']))
    return redirect(url_for('index'))
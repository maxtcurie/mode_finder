from scipy import optimize
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import tkinter as tk


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2)


def gaussian_fit_GUI(x,data):
    x,data=np.array(x), np.array(data)

    global amplitude
    global mean
    global stddev


    root=tk.Tk()
    root.title('Gaussian Fit')
    #root.geometry("500x500")
    
    #load the icon for the GUI 
    root.iconbitmap('./Physics_helper_logo.ico')

    #warnings.simplefilter("error", OptimizeWarning)
    judge=0

    frame_plot=tk.LabelFrame(root, text='Plot of the data and fitting',padx=20,pady=20)
    frame_plot.grid(row=0,column=0)
    fig = Figure(figsize = (5, 5),
                dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.plot(x,data, label="data")

    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  
        max_index=np.argmax(data)

        amplitude=popt[0]
        mean=popt[1]
        stddev=popt[2]

        plot1.plot(x, gaussian(x, *popt), label="fit")
        plot1.axvline(mean,color='red',alpha=0.5)
        plot1.axvline(mean+stddev,color='red',alpha=0.5)
        plot1.axvline(mean-stddev,color='red',alpha=0.5)
        plot1.legend()

    except RuntimeError:
        print("Curve fit failed, need to fit manually")
        pass  

    canvas = FigureCanvasTkAgg(fig,master = frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,frame_plot)
    toolbar.update()
    canvas.get_tk_widget().pack()

    root1=tk.LabelFrame(root, text='User Box',padx=20,pady=20)
    root1.grid(row=0,column=1)


    frame1=tk.LabelFrame(root1, text='Accept/Reject the fit',padx=20,pady=20)
    frame1.grid(row=0,column=0)
    #import an option button
    opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()

    option_button11=tk.Radiobutton(frame1, text='Accept the fit',\
                                variable=opt_var1, value=1,\
                                command=lambda: click_button(opt_var1.get(), root,root1))
    option_button11.grid(row=1,column=0)

    option_button12=tk.Radiobutton(frame1, text='Enter manually',\
                                variable=opt_var1, value=2,\
                                command=lambda: click_button(opt_var1.get(), root,root1))
    option_button12.grid(row=2,column=0)
        

    

    def click_button(value, root, root1):

        global amplitude
        global mean
        global stddev 

        label_list=[]

        frame_data=tk.LabelFrame(root1, text='Current fitting parameter',padx=80,pady=20)
        frame_data.grid(row=1,column=0)
        label_list.append(frame_data)

        amplitude_string=f'amplitude = {amplitude}'
        mean_string     =f'mean      = {mean}'
        std_string      =f'stddev    = {stddev}'
        amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
        mean_string=mean_string+' '*(50-len(mean_string))
        std_string=std_string+' '*(50-len(std_string))

        amp_label =tk.Label(frame_data,text=amplitude_string)
        mean_label=tk.Label(frame_data,text=mean_string)
        std_label =tk.Label(frame_data,text=std_string)
        amp_label.grid(row=0,column=0)
        mean_label.grid(row=1,column=0)
        std_label.grid(row=2,column=0)
        label_list.append(amp_label)
        label_list.append(mean_label)
        label_list.append(std_label)

        def plot_manual_fit(x,data,mean_Input,sigma_Input,root,label_list):
            #frame_plot.grid_forget()

            frame_plot=tk.LabelFrame(root, text='Plot of the data and fitting',padx=20,pady=20)
            frame_plot.grid(row=0,column=0)
            fig = Figure(figsize = (5, 5),
                        dpi = 100)
            plot1 = fig.add_subplot(111)
            plot1.plot(x,data, label="data")
            
            global amplitude
            global mean
            global stddev 

            mean=float(mean_Input)
            amplitude=data[np.argmin(abs(x-mean))]
            stddev=float(sigma_Input)


            frame_data=label_list[0]
            amp_label=label_list[1]
            mean_label=label_list[2]
            std_label=label_list[3]

            amp_label.grid_forget()
            mean_label.grid_forget()
            std_label.grid_forget()

            amplitude_string=f'amplitude = {amplitude}'
            mean_string     =f'mean      = {mean}'
            std_string      =f'stddev    = {stddev}'

            amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
            mean_string=mean_string+' '*(50-len(mean_string))
            std_string=std_string+' '*(50-len(std_string))


            amp_label =tk.Label(frame_data,text=amplitude_string)
            mean_label=tk.Label(frame_data,text=mean_string)
            std_label =tk.Label(frame_data,text=std_string)
            amp_label.grid(row=0,column=0)
            mean_label.grid(row=1,column=0)
            std_label.grid(row=2,column=0)

            plot1.plot(x, gaussian(x, amplitude, mean, stddev), label="fit")
            plot1.axvline(mean,color='red',alpha=0.5)
            plot1.axvline(mean+stddev,color='red',alpha=0.5)
            plot1.axvline(mean-stddev,color='red',alpha=0.5)
            plot1.legend()
        
            canvas = FigureCanvasTkAgg(fig,master = frame_plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas,frame_plot)
            toolbar.update()
            canvas.get_tk_widget().pack()

        if value==1:
            #'Accept the fit'
            myButton2=tk.Button(root1, text='Save and Continue', command=root.quit)
            myButton2.grid(row=3,column=0)

        elif value==2:
            #Label1.grid_forget()
            #'Enter manually'
            frame_input=tk.LabelFrame(root1, text='Manual Input',padx=10,pady=10)
            frame_input.grid(row=2,column=0)
            tk.Label(frame_input,text='mu(center) = ').grid(row=0,column=0)
            mean_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            mean_Input_box.insert(0,'')
            mean_Input_box.grid(row=0,column=1)
        
            tk.Label(frame_input,text='sigma(spread) = ').grid(row=1,column=0)
            sigma_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            sigma_Input_box.insert(0,'')
            sigma_Input_box.grid(row=1,column=1)

            

            Plot_Button=tk.Button(frame_input, text='Plot the Manual Fit',\
                command=lambda: plot_manual_fit(x,data,\
                    float( mean_Input_box.get()  ),\
                    float( sigma_Input_box.get() ),\
                    root,label_list\
                    )  \
                )
            Plot_Button.grid(row=2,column=1)

            Save_Button=tk.Button(root1, text='Save and Continue',\
                     state=tk.DISABLED)#state: tk.DISABLED, or tk.NORMAL
            Save_Button.grid(row=3,column=0)

    #creat the GUI
    root.mainloop()

    return amplitude,mean,stddev


x=np.arange(-5,5,0.01)
y=np.exp(-x**2.)

amplitude,mean,stddev=gaussian_fit_GUI(x,y)
print(f'amplitude,mean,stddev = {amplitude}, {mean}, {stddev}')
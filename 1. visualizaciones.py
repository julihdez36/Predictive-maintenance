import matplotlib.pyplot as plt

# <---Fake Data for Plotting---->
# Median ages 
ages = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

# Median Microbiologist Salaries by Age
mib_salary = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]

# Median Pharmacist Salaries by Age
pharma_salary = [45372, 48876, 53850, 57287, 63016,65998, 70003, 70000, 71496, 75370, 83640]

# Median Cader Salaries by Age
bcs_salary = [37810, 43515, 46823, 49293, 53437,56373, 62375, 66674, 68745, 68746, 74583]

plt.plot(ages, mib_salary)
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.show() 


### Comparison ###
# Microbiology
plt.plot(ages, mib_salary)
# Pharmacy
plt.plot(ages, pharma_salary)
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.show() 

### Add Legend ###


# Microbiology
plt.plot(ages, mib_salary, label="Microbiology")
# Pharmacy
plt.plot(ages, pharma_salary, label= "Pharmacy")
# BCS
plt.plot(ages, bcs_salary, label= "BCS")

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.legend() 
plt.show()

# Nicely fit in the figure

# Microbiology
plt.plot(ages, mib_salary, label="Microbiology")
# Pharmacy
plt.plot(ages, pharma_salary, label= "Pharmacy")
# BCS
plt.plot(ages, bcs_salary, label= "BCS")

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.legend() 
plt.tight_layout()
plt.show()  

# Customization

# Microbiology
plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')
# Pharmacy
plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')
# BCS
plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')

plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.legend() 
plt.tight_layout()
plt.show() 
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data (or from HDF5 file):
# - 3 of jumping and 3 of walking
# - plot all of them along acceleration vs time

# Time: Index 0, Absolute acceleration: Index 4

sam_jumping = pd.read_csv("./Sam_Raw/Front-coat-jump.csv").values
sam_walking = pd.read_csv("./Sam_Raw/Left-pocket.csv").values

john_jumping = pd.read_csv("./John_Raw/Front Coat Pocket Jump.csv").values
john_walking = pd.read_csv("./John_Raw/Left Pocket Walk.csv").values

vivian_jumping = pd.read_csv("./Vivian_raw/Coat Chest Jumping.csv").values
vivian_walking = pd.read_csv("./Vivian_raw/Left Coat Walking.csv").values


# Plot time against acceleration (columns 0 and 4 of the csv files)

# plotting x against time
# plt.plot(sam_jumping[:5000, 0], sam_jumping[:5000, 1])

# plotting y against time
# plt.plot(sam_jumping[:5000, 0], sam_jumping[:5000, 2])

# # plotting z against time
# plt.plot(sam_jumping[:5000, 0], sam_jumping[:5000, 3])
# plt.xlabel("time")
# plt.ylabel("acceleration (m/s^2)")
# plt.show()


# plt.plot(john_jumping[:5000, 0], john_jumping[:5000, 4])
# plt.plot(john_walking[:5000, 0], john_walking[:5000, 4])
# plt.xlabel("time")
# plt.ylabel("acceleration (m/s^2)")
# plt.show()

##### JUMPING #####
plt.plot(vivian_jumping[:5000, 0], vivian_jumping[:5000, 1])  # X acceleration
plt.xlabel("time (s)")
plt.ylabel("X-acceleration (m/s^2)")
plt.show()

plt.plot(vivian_jumping[:5000, 0], vivian_jumping[:5000, 2])  # Y acceleration
plt.xlabel("time (s)")
plt.ylabel("Y-acceleration (m/s^2)")
plt.show()

plt.plot(vivian_jumping[:5000, 0], vivian_jumping[:5000, 3])  # Z acceleration
plt.xlabel("time (s)")
plt.ylabel("Z-acceleration (m/s^2)")
plt.show()


##### WALKING #####
plt.plot(vivian_walking[:5000, 0], vivian_walking[:5000, 1])  # X accelerration
plt.xlabel("time (s)")
plt.ylabel("X-acceleration (m/s^2)")
plt.show()

plt.plot(vivian_walking[:5000, 0], vivian_walking[:5000, 2])  # Y accelerration
plt.xlabel("time (s)")
plt.ylabel("Y-acceleration (m/s^2)")
plt.show()

plt.plot(vivian_walking[:5000, 0], vivian_walking[:5000, 3])  # Z accelerration
plt.xlabel("time (s)")
plt.ylabel("Z-acceleration (m/s^2)")
plt.show()

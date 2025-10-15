# Graph ML

### Dataset setup

<pre>
# Install git-lfs
brew install git-lfs

# or
apt-get update 
apt-get install git-lfs

# initialize LFS
git lfs install

# pull large files
git lfs pull
 </pre>

---

 ### RunPod Cluster Setup

 **Steps:**
 1. Go to RunPod.
 2. In the top left corner (where your username appears) open the dropdown and select "Franek's Team" (if you accepted my invite you will see it).
 3. On the side bar below you will see a tab called "Pods".
 4. Open the tab and at the top of the page, just under the "Select an Instance" text, select the storage volume called "GraphML".
 5. Then click on any of the available nodes on that page.
 6. You should see a purple "Change Template" button. Click that and select the "graph_ml_updated" template.
 7. Finally click the big "Deploy On-Demand" button.
 8. The pod will be set up with all the dependencies and on startup credentails for our git repo will be configured - they are stored in teh network volume and will just get initialised.
 9. Once the pod builds, you should see a "jupyterlab" button. Click that and jupyter will open in your browser. You can acess the terminal and the storage volume from there.
 10. You should be read to go.

 **Also please be sure NOT to delte the storage volume, and remember to terminate the pod once you are done using it!!! Otherwise it will be idling and burning through funds.**
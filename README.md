# DeblurCNN
Convolutional Neural Network to deblur the images. A 20 layer DnCNN trained to remove blur introduced by averaging filter.

## Architecture of DnCNN used
<table bgcolor:"red">
 <tr>
  <td>Convolution Layer - 1 (3 x 3 kernel) (Input Channels = 3) (Output Channels = 64)</td>
 </tr>
 <tr>
  <td>Activation (ReLU)</td>
 </tr>
 <tr>
 <td>Convolution Layer - 2 (3 x 3 kernel) (Input Channels = 64) (Output Channels = 64)</td>
 </tr>
 <tr>
  <td>Batch Normalization</td>
 </tr>
 <tr>
  <td>Activation (ReLU)</td>
 </tr>
 <tr>
 <td>Convolution Layer - 3 (3 x 3 kernel) (Input Channels = 64) (Output Channels = 64)</td>
 </tr>
 <tr>
  <td>Batch Normalization</td>
 </tr>
 <tr>
  <td>Activation (ReLU)</td>
 </tr>
 <tr>
<td>Convolution Layer - 4 (3 x 3 kernel) (Input Channels = 64) (Output Channels = 64)</td>
 </tr>
 <tr>
  <td>Batch Normalization</td>
 </tr>
 <tr>
  <td>Activation (ReLU)</td>
 </tr>
<td>...</td>
 </tr>
 <tr>
  <td>...</td>
 </tr>
 <tr>
  <td>...</td>
 </tr>
<td>...</td>
 </tr>
 <tr>
  <td>...</td>
 </tr>
 <tr>
  <td>...</td>
 </tr>
 <tr>
 <td>Convolution Layer - 20 (3 x 3 kernel) (Input Channels = 64) (Output Channels = 3)</td>
 </tr>
 </table>
 
## Some Results

<table>
 <tr>
  <th>Original Image</th>
  <th>Blurred Image (Input to DeblurCNN)</th>
  <th>Deblurred Image (Output of DeblurCNN)</th>
 </tr>
 <tr>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Q1.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Qb1.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Y1.jpg" width="100%" height="100%"></td>
 </tr>
 <tr>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Q2.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Qb2.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Y2.jpg" width="100%" height="100%"></td>
 </tr>
 <tr>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Q3.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Qb3.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Y3.jpg" width="100%" height="100%"></td>
 </tr>
 <tr>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Q4.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Qb4.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Y4.jpg" width="100%" height="100%"></td>
 </tr>
 <tr>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Q5.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Qb5.jpg" width="100%" height="100%"></td>
  <td><img src="https://github.com/rohit-pardasani/DeblurCNN/blob/master/MyDatasetTest/Y5.jpg" width="100%" height="100%"></td>
 </tr>
</table>




мн*
Д”
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ЉГ(
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

:*
dtype0
Ж
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/v/dense_1/kernel
А
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	А*
dtype0
З
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/m/dense_1/kernel
А
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	А*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:А*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:А*
dtype0
Д
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†А*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
†А*
dtype0
Д
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†А*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
†А*
dtype0
М
Adam/v/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/v/lstm/lstm_cell/bias
Е
.Adam/v/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0
М
Adam/m/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/m/lstm/lstm_cell/bias
Е
.Adam/m/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0
®
&Adam/v/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/v/lstm/lstm_cell/recurrent_kernel
°
:Adam/v/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/v/lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
®
&Adam/m/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/m/lstm/lstm_cell/recurrent_kernel
°
:Adam/m/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/m/lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
Ф
Adam/v/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameAdam/v/lstm/lstm_cell/kernel
Н
0Adam/v/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/kernel*
_output_shapes

:@*
dtype0
Ф
Adam/m/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameAdam/m/lstm/lstm_cell/kernel
Н
0Adam/m/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/kernel*
_output_shapes

:@*
dtype0
Ф
Adam/v/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
—Ж*,
shared_nameAdam/v/embedding/embeddings
Н
/Adam/v/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding/embeddings* 
_output_shapes
:
—Ж*
dtype0
Ф
Adam/m/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
—Ж*,
shared_nameAdam/m/embedding/embeddings
Н
/Adam/m/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding/embeddings* 
_output_shapes
:
—Ж*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
~
lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namelstm/lstm_cell/bias
w
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes
:@*
dtype0
Ъ
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!lstm/lstm_cell/recurrent_kernel
У
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
Ж
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†А*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
†А*
dtype0
Ж
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
—Ж*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
—Ж*
dtype0
В
serving_default_embedding_inputPlaceholder*'
_output_shapes
:€€€€€€€€€2*
dtype0*
shape:€€€€€€€€€2
Ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingslstm/lstm_cell/kernellstm/lstm_cell/biaslstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_17144

NoOpNoOp
шZ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≥Z
value©ZB¶Z BЯZ
√
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
†
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
•
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator* 
Ѕ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec*
О
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
¶
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
•
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator* 
¶
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
•
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator* 
¶
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias*
J
0
V1
W2
X3
64
75
E6
F7
T8
U9*
J
0
V1
W2
X3
64
75
E6
F7
T8
U9*
%
Y0
Z1
[2
\3
]4* 
∞
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
6
gtrace_0
htrace_1
itrace_2
jtrace_3* 
* 
Б
k
_variables
l_iterations
m_learning_rate
n_index_dict
o
_momentums
p_velocities
q_update_step_xla*

rserving_default* 

0*

0*
	
Y0* 
У
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0
Аtrace_1* 

Бtrace_0
Вtrace_1* 
* 

V0
W1
X2*

V0
W1
X2*

Г0
Д1* 
•
Еstates
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
:
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_3* 
* 
л
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щ_random_generator
Ъ
state_size

Vkernel
Wrecurrent_kernel
Xbias*
* 
* 
* 
* 
Ц
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

†trace_0* 

°trace_0* 

60
71*

60
71*

Z0
[1* 
Ш
Ґnon_trainable_variables
£layers
§metrics
 •layer_regularization_losses
¶layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Іtrace_0* 

®trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Ѓtrace_0
ѓtrace_1* 

∞trace_0
±trace_1* 
* 

E0
F1*

E0
F1*

\0
]1* 
Ш
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

Јtrace_0* 

Єtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

Њtrace_0
њtrace_1* 

јtrace_0
Ѕtrace_1* 
* 

T0
U1*

T0
U1*
* 
Ш
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

«trace_0* 

»trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

…trace_0* 

 trace_0* 

Ћtrace_0* 

ћtrace_0* 

Ќtrace_0* 
* 
C
0
1
2
3
4
5
6
7
	8*

ќ0
ѕ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
ґ
l0
–1
—2
“3
”4
‘5
’6
÷7
„8
Ў9
ў10
Џ11
џ12
№13
Ё14
ё15
я16
а17
б18
в19
г20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
–0
“1
‘2
÷3
Ў4
Џ5
№6
ё7
а8
в9*
T
—0
”1
’2
„3
ў4
џ5
Ё6
я7
б8
г9*
* 
* 
* 
* 
* 
	
Y0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

дtrace_0* 

еtrace_0* 
* 
* 

(0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1
X2*

V0
W1
X2*

Г0
Д1* 
Ю
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses*

лtrace_0
мtrace_1* 

нtrace_0
оtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Z0
[1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
п	variables
р	keras_api

сtotal

тcount*
M
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs*
f`
VARIABLE_VALUEAdam/m/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/lstm/lstm_cell/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/lstm/lstm_cell/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/lstm/lstm_cell/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/lstm/lstm_cell/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

Г0
Д1* 
* 
* 
* 
* 
* 

с0
т1*

п	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

х0
ц1*

у	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
€
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp/Adam/m/embedding/embeddings/Read/ReadVariableOp/Adam/v/embedding/embeddings/Read/ReadVariableOp0Adam/m/lstm/lstm_cell/kernel/Read/ReadVariableOp0Adam/v/lstm/lstm_cell/kernel/Read/ReadVariableOp:Adam/m/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp:Adam/v/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp.Adam/m/lstm/lstm_cell/bias/Read/ReadVariableOp.Adam/v/lstm/lstm_cell/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOp)Adam/m/dense_2/kernel/Read/ReadVariableOp)Adam/v/dense_2/kernel/Read/ReadVariableOp'Adam/m/dense_2/bias/Read/ReadVariableOp'Adam/v/dense_2/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_19950
™
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcount*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_20068В≈&
СD
б
?__inference_lstm_layer_call_and_return_conditional_losses_15875

inputs!
lstm_cell_15785:@
lstm_cell_15787:@!
lstm_cell_15789:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskв
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15785lstm_cell_15787lstm_cell_15789*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15739n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15785lstm_cell_15787lstm_cell_15789*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_15798*
condR
while_cond_15797*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    З
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOplstm_cell_15785*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOplstm_cell_15787*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€д
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
УЊ
щ
while_body_19123
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ђ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ”
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
УЊ
щ
while_body_16583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ђ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ”
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Љ	
Ґ
lstm_while_cond_17337&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_17337___redundant_placeholder0=
9lstm_while_lstm_while_cond_17337___redundant_placeholder1=
9lstm_while_lstm_while_cond_17337___redundant_placeholder2=
9lstm_while_lstm_while_cond_17337___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
пP
Ц
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19665

inputs
states_0
states_1/
split_readvariableop_resource:@-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ю
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:€€€€€€€€€S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: С
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
У	
ѓ
__inference_loss_fn_3_19514L
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:	А
identityИҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpЂ
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp
Г÷
—
E__inference_sequential_layer_call_and_return_conditional_losses_17525

inputs4
 embedding_embedding_lookup_17226:
—Ж>
,lstm_lstm_cell_split_readvariableop_resource:@<
.lstm_lstm_cell_split_1_readvariableop_resource:@8
&lstm_lstm_cell_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
†А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐembedding/embedding_lookupҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/while_
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2б
embedding/embedding_lookupResourceGather embedding_embedding_lookup_17226embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/17226*+
_output_shapes
:€€€€€€€€€2*
dtype0њ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/17226*+
_output_shapes
:€€€€€€€€€2Х
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2В
dropout/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€2S

lstm/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          К
lstm/transpose	Transposedropout/Identity:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€√
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Л
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   п
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskk
lstm/lstm_cell/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:c
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?§
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
 lstm/lstm_cell/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:e
 lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?™
lstm/lstm_cell/ones_like_1Fill)lstm/lstm_cell/ones_like_1/Shape:output:0)lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Н
lstm/lstm_cell/mulMullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
lstm/lstm_cell/mul_1Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
lstm/lstm_cell/mul_2Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
lstm/lstm_cell/mul_3Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Р
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitИ
lstm/lstm_cell/MatMulMatMullstm/lstm_cell/mul:z:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_1MatMullstm/lstm_cell/mul_1:z:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_2MatMullstm/lstm_cell/mul_2:z:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_3MatMullstm/lstm_cell/mul_3:z:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Р
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ѕ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitХ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_4Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_5Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_6Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_7Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ґ
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskФ
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul_4:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€С
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_5:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm/lstm_cell/mul_8Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_6:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm/lstm_cell/mul_9Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Г
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_8:z:0lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_7:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€i
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_10Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   «
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ≥

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_17338*!
condR
lstm_while_cond_17337*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Ж
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   —
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:†
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          •
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   {
flatten/ReshapeReshapelstm/transpose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†В
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аk
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0О
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOp embedding_embedding_lookup_17226* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: §
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: †
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ш
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding/embedding_lookup7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
н
¶
__inference_loss_fn_4_19523E
7dense_1_bias_regularizer_l2loss_readvariableop_resource:
identityИҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp
СD
б
?__inference_lstm_layer_call_and_return_conditional_losses_15554

inputs!
lstm_cell_15464:@
lstm_cell_15466:@!
lstm_cell_15468:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐ!lstm_cell/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskв
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15464lstm_cell_15466lstm_cell_15468*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15463n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15464lstm_cell_15466lstm_cell_15468*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_15477*
condR
while_cond_15476*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    З
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOplstm_cell_15464*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOplstm_cell_15466*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€д
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ёR
Щ
E__inference_sequential_layer_call_and_return_conditional_losses_16917

inputs#
embedding_16859:
—Ж

lstm_16863:@

lstm_16865:@

lstm_16867:@
dense_16871:
†А
dense_16873:	А 
dense_1_16877:	А
dense_1_16879:
dense_2_16883:
dense_2_16885:
identityИҐdense/StatefulPartitionedCallҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/StatefulPartitionedCallҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/StatefulPartitionedCallҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpв
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_16859*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_15905н
dropout/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_16818С
lstm/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
lstm_16863
lstm_16865
lstm_16867*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16789’
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_16180ь
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16871dense_16873*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_16201М
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16378Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_16877dense_1_16879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_16233П
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16345Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_16883dense_2_16885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_16257И
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_16859* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16863*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16865*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16871* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16873*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16877*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16879*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€∆
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/StatefulPartitionedCall6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ш

у
B__inference_dense_2_layer_call_and_return_conditional_losses_16257

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
п
)__inference_lstm_cell_layer_call_fn_19575

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
РК
Ј
?__inference_lstm_layer_call_and_return_conditional_losses_16166

inputs9
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_16024*
condR
while_cond_16023*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Гћ
ќ

 __inference__wrapped_model_15338
embedding_input?
+sequential_embedding_embedding_lookup_15067:
—ЖI
7sequential_lstm_lstm_cell_split_readvariableop_resource:@G
9sequential_lstm_lstm_cell_split_1_readvariableop_resource:@C
1sequential_lstm_lstm_cell_readvariableop_resource:@C
/sequential_dense_matmul_readvariableop_resource:
†А?
0sequential_dense_biasadd_readvariableop_resource:	АD
1sequential_dense_1_matmul_readvariableop_resource:	А@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identityИҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpҐ)sequential/dense_2/BiasAdd/ReadVariableOpҐ(sequential/dense_2/MatMul/ReadVariableOpҐ%sequential/embedding/embedding_lookupҐ(sequential/lstm/lstm_cell/ReadVariableOpҐ*sequential/lstm/lstm_cell/ReadVariableOp_1Ґ*sequential/lstm/lstm_cell/ReadVariableOp_2Ґ*sequential/lstm/lstm_cell/ReadVariableOp_3Ґ.sequential/lstm/lstm_cell/split/ReadVariableOpҐ0sequential/lstm/lstm_cell/split_1/ReadVariableOpҐsequential/lstm/whiles
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2Н
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_15067sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/15067*+
_output_shapes
:€€€€€€€€€2*
dtype0а
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/15067*+
_output_shapes
:€€€€€€€€€2Ђ
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2Ш
sequential/dropout/IdentityIdentity9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€2i
sequential/lstm/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ґ
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ђ
sequential/lstm/transpose	Transpose$sequential/dropout/Identity:output:0'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€d
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:o
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€д
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ц
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Р
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“o
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskБ
)sequential/lstm/lstm_cell/ones_like/ShapeShape(sequential/lstm/strided_slice_2:output:0*
T0*
_output_shapes
:n
)sequential/lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≈
#sequential/lstm/lstm_cell/ones_likeFill2sequential/lstm/lstm_cell/ones_like/Shape:output:02sequential/lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
+sequential/lstm/lstm_cell/ones_like_1/ShapeShapesequential/lstm/zeros:output:0*
T0*
_output_shapes
:p
+sequential/lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ћ
%sequential/lstm/lstm_cell/ones_like_1Fill4sequential/lstm/lstm_cell/ones_like_1/Shape:output:04sequential/lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ѓ
sequential/lstm/lstm_cell/mulMul(sequential/lstm/strided_slice_2:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€∞
sequential/lstm/lstm_cell/mul_1Mul(sequential/lstm/strided_slice_2:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€∞
sequential/lstm/lstm_cell/mul_2Mul(sequential/lstm/strided_slice_2:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€∞
sequential/lstm/lstm_cell/mul_3Mul(sequential/lstm/strided_slice_2:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
.sequential/lstm/lstm_cell/split/ReadVariableOpReadVariableOp7sequential_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0м
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:06sequential/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split©
 sequential/lstm/lstm_cell/MatMulMatMul!sequential/lstm/lstm_cell/mul:z:0(sequential/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
"sequential/lstm/lstm_cell/MatMul_1MatMul#sequential/lstm/lstm_cell/mul_1:z:0(sequential/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€≠
"sequential/lstm/lstm_cell/MatMul_2MatMul#sequential/lstm/lstm_cell/mul_2:z:0(sequential/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€≠
"sequential/lstm/lstm_cell/MatMul_3MatMul#sequential/lstm/lstm_cell/mul_3:z:0(sequential/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€m
+sequential/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¶
0sequential/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0в
!sequential/lstm/lstm_cell/split_1Split4sequential/lstm/lstm_cell/split_1/split_dim:output:08sequential/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitґ
!sequential/lstm/lstm_cell/BiasAddBiasAdd*sequential/lstm/lstm_cell/MatMul:product:0*sequential/lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
#sequential/lstm/lstm_cell/BiasAdd_1BiasAdd,sequential/lstm/lstm_cell/MatMul_1:product:0*sequential/lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ї
#sequential/lstm/lstm_cell/BiasAdd_2BiasAdd,sequential/lstm/lstm_cell/MatMul_2:product:0*sequential/lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ї
#sequential/lstm/lstm_cell/BiasAdd_3BiasAdd,sequential/lstm/lstm_cell/MatMul_3:product:0*sequential/lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€®
sequential/lstm/lstm_cell/mul_4Mulsequential/lstm/zeros:output:0.sequential/lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
sequential/lstm/lstm_cell/mul_5Mulsequential/lstm/zeros:output:0.sequential/lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
sequential/lstm/lstm_cell/mul_6Mulsequential/lstm/zeros:output:0.sequential/lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
sequential/lstm/lstm_cell/mul_7Mulsequential/lstm/zeros:output:0.sequential/lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
(sequential/lstm/lstm_cell/ReadVariableOpReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0~
-sequential/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        А
/sequential/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       А
/sequential/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
'sequential/lstm/lstm_cell/strided_sliceStridedSlice0sequential/lstm/lstm_cell/ReadVariableOp:value:06sequential/lstm/lstm_cell/strided_slice/stack:output:08sequential/lstm/lstm_cell/strided_slice/stack_1:output:08sequential/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskµ
"sequential/lstm/lstm_cell/MatMul_4MatMul#sequential/lstm/lstm_cell/mul_4:z:00sequential/lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€≤
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/BiasAdd:output:0,sequential/lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€Б
!sequential/lstm/lstm_cell/SigmoidSigmoid!sequential/lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
*sequential/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       В
1sequential/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        В
1sequential/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_1StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_1:value:08sequential/lstm/lstm_cell/strided_slice_1/stack:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_5MatMul#sequential/lstm/lstm_cell/mul_5:z:02sequential/lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
sequential/lstm/lstm_cell/add_1AddV2,sequential/lstm/lstm_cell/BiasAdd_1:output:0,sequential/lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€Е
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid#sequential/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€£
sequential/lstm/lstm_cell/mul_8Mul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
*sequential/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        В
1sequential/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   В
1sequential/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_2StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_2:value:08sequential/lstm/lstm_cell/strided_slice_2/stack:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_6MatMul#sequential/lstm/lstm_cell/mul_6:z:02sequential/lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
sequential/lstm/lstm_cell/add_2AddV2,sequential/lstm/lstm_cell/BiasAdd_2:output:0,sequential/lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€}
sequential/lstm/lstm_cell/TanhTanh#sequential/lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€£
sequential/lstm/lstm_cell/mul_9Mul%sequential/lstm/lstm_cell/Sigmoid:y:0"sequential/lstm/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€§
sequential/lstm/lstm_cell/add_3AddV2#sequential/lstm/lstm_cell/mul_8:z:0#sequential/lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
*sequential/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   В
1sequential/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        В
1sequential/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_3StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_3:value:08sequential/lstm/lstm_cell/strided_slice_3/stack:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_7MatMul#sequential/lstm/lstm_cell/mul_7:z:02sequential/lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
sequential/lstm/lstm_cell/add_4AddV2,sequential/lstm/lstm_cell/BiasAdd_3:output:0,sequential/lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€Е
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid#sequential/lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€
 sequential/lstm/lstm_cell/Tanh_1Tanh#sequential/lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€®
 sequential/lstm/lstm_cell/mul_10Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0$sequential/lstm/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   и
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“V
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€d
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ќ
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_lstm_lstm_cell_split_readvariableop_resource9sequential_lstm_lstm_cell_split_1_readvariableop_resource1sequential_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 sequential_lstm_while_body_15179*,
cond$R"
 sequential_lstm_while_cond_15178*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations С
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   т
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0x
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€q
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:„
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ∆
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2k
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ь
sequential/flatten/ReshapeReshapesequential/lstm/transpose_1:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Ш
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АБ
sequential/dropout_1/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АЫ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0ѓ
sequential/dense_1/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
sequential/dropout_2/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ѓ
sequential/dense_2/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€m
IdentityIdentitysequential/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Э
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup)^sequential/lstm/lstm_cell/ReadVariableOp+^sequential/lstm/lstm_cell/ReadVariableOp_1+^sequential/lstm/lstm_cell/ReadVariableOp_2+^sequential/lstm/lstm_cell/ReadVariableOp_3/^sequential/lstm/lstm_cell/split/ReadVariableOp1^sequential/lstm/lstm_cell/split_1/ReadVariableOp^sequential/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2T
(sequential/lstm/lstm_cell/ReadVariableOp(sequential/lstm/lstm_cell/ReadVariableOp2X
*sequential/lstm/lstm_cell/ReadVariableOp_1*sequential/lstm/lstm_cell/ReadVariableOp_12X
*sequential/lstm/lstm_cell/ReadVariableOp_2*sequential/lstm/lstm_cell/ReadVariableOp_22X
*sequential/lstm/lstm_cell/ReadVariableOp_3*sequential/lstm/lstm_cell/ReadVariableOp_32`
.sequential/lstm/lstm_cell/split/ReadVariableOp.sequential/lstm/lstm_cell/split/ReadVariableOp2d
0sequential/lstm/lstm_cell/split_1/ReadVariableOp0sequential/lstm/lstm_cell/split_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
Н

ы
#__inference_signature_wrapper_17144
embedding_input
unknown:
—Ж
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
†А
	unknown_4:	А
	unknown_5:	А
	unknown_6:
	unknown_7:
	unknown_8:
identityИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_15338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
с
ю
 sequential_lstm_while_cond_15178<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1S
Osequential_lstm_while_sequential_lstm_while_cond_15178___redundant_placeholder0S
Osequential_lstm_while_sequential_lstm_while_cond_15178___redundant_placeholder1S
Osequential_lstm_while_sequential_lstm_while_cond_15178___redundant_placeholder2S
Osequential_lstm_while_sequential_lstm_while_cond_15178___redundant_placeholder3"
sequential_lstm_while_identity
Ґ
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: k
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ю
∞
$__inference_lstm_layer_call_fn_18047
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_15875|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
УЊ
щ
while_body_18493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ђ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ”
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?¶
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:∞
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>№
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    џ
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€≠
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
З"
Ѕ
while_body_15477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_lstm_cell_15501_0:@%
while_lstm_cell_15503_0:@)
while_lstm_cell_15505_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_lstm_cell_15501:@#
while_lstm_cell_15503:@'
while_lstm_cell_15505:@ИҐ'while/lstm_cell/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0†
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15501_0while_lstm_cell_15503_0while_lstm_cell_15505_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15463ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Н
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Н
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_15501while_lstm_cell_15501_0"0
while_lstm_cell_15503while_lstm_cell_15503_0"0
while_lstm_cell_15505while_lstm_cell_15505_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ѕ
Х
'__inference_dense_1_layer_call_fn_19412

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_16233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
‘
@__inference_dense_layer_call_and_return_conditional_losses_16201

inputs2
matmul_readvariableop_resource:
†А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АП
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А„
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Ю
∞
$__inference_lstm_layer_call_fn_18036
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_15554|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
є

В
*__inference_sequential_layer_call_fn_16965
embedding_input
unknown:
—Ж
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
†А
	unknown_4:	А
	unknown_5:	А
	unknown_6:
	unknown_7:
	unknown_8:
identityИҐStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_16917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
ф	
Љ
__inference_loss_fn_5_19532R
@lstm_lstm_cell_kernel_regularizer_l2loss_readvariableop_resource:@
identityИҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpЄ
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@lstm_lstm_cell_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)lstm/lstm_cell/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: А
NoOpNoOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp
∞
Њ
while_cond_18807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18807___redundant_placeholder03
/while_while_cond_18807___redundant_placeholder13
/while_while_cond_18807___redundant_placeholder23
/while_while_cond_18807___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ю

щ
*__inference_sequential_layer_call_fn_17222

inputs
unknown:
—Ж
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
†А
	unknown_4:	А
	unknown_5:	А
	unknown_6:
	unknown_7:
	unknown_8:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_16917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
е
`
B__inference_dropout_layer_call_and_return_conditional_losses_18013

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€2_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ш

у
B__inference_dense_2_layer_call_and_return_conditional_losses_19478

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ	
і
__inference_loss_fn_6_19541L
>lstm_lstm_cell_bias_regularizer_l2loss_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp∞
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp>lstm_lstm_cell_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'lstm/lstm_cell/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp
й	
Љ
__inference_loss_fn_0_19487S
?embedding_embeddings_regularizer_l2loss_readvariableop_resource:
—Ж
identityИҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpЄ
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOp?embedding_embeddings_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: f
IdentityIdentity(embedding/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp
З—
є
?__inference_lstm_layer_call_and_return_conditional_losses_18699
inputs_09
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:†
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ї
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Е
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18493*
condR
while_cond_18492*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
∞
Њ
while_cond_16582
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16582___redundant_placeholder03
/while_while_cond_16582___redundant_placeholder13
/while_while_cond_16582___redundant_placeholder23
/while_while_cond_16582___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
∞
Њ
while_cond_16023
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16023___redundant_placeholder03
/while_while_cond_16023___redundant_placeholder13
/while_while_cond_16023___redundant_placeholder23
/while_while_cond_16023___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Т|
®

lstm_while_body_17338&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0F
4lstm_while_lstm_cell_split_readvariableop_resource_0:@D
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:@@
.lstm_while_lstm_cell_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorD
2lstm_while_lstm_cell_split_readvariableop_resource:@B
4lstm_while_lstm_cell_split_1_readvariableop_resource:@>
,lstm_while_lstm_cell_readvariableop_resource:@ИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpН
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   њ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Й
$lstm/while/lstm_cell/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:i
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
&lstm/while/lstm_cell/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:k
&lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Љ
 lstm/while/lstm_cell/ones_like_1Fill/lstm/while/lstm_cell/ones_like_1/Shape:output:0/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€±
lstm/while/lstm_cell/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€≥
lstm/while/lstm_cell/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€≥
lstm/while/lstm_cell/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€≥
lstm/while/lstm_cell/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ё
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЪ
lstm/while/lstm_cell/MatMulMatMullstm/while/lstm_cell/mul:z:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_1MatMullstm/while/lstm_cell/mul_1:z:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_2MatMullstm/while/lstm_cell/mul_2:z:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_3MatMullstm/while/lstm_cell/mul_3:z:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ю
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0”
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitІ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/while/lstm_cell/mul_4Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/while/lstm_cell/mul_5Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/while/lstm_cell/mul_6Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/while/lstm_cell/mul_7Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Т
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¶
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul_4:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€£
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_5:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€С
lstm/while/lstm_cell/mul_8Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_6:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
lstm/while/lstm_cell/mul_9Mul lstm/while/lstm_cell/Sigmoid:y:0lstm/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_8:z:0lstm/while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_7:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€u
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/while/lstm_cell/mul_10Mul"lstm/while/lstm_cell/Sigmoid_2:y:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€„
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: Х
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: Ж
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_10:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€…
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
√N
Є
E__inference_sequential_layer_call_and_return_conditional_losses_17026
embedding_input#
embedding_16968:
—Ж

lstm_16972:@

lstm_16974:@

lstm_16976:@
dense_16980:
†А
dense_16982:	А 
dense_1_16986:	А
dense_1_16988:
dense_2_16992:
dense_2_16994:
identityИҐdense/StatefulPartitionedCallҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/StatefulPartitionedCallҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/StatefulPartitionedCallҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpл
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_16968*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_15905Ё
dropout/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_15914Й
lstm/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
lstm_16972
lstm_16974
lstm_16976*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16166’
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_16180ь
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16980dense_16982*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_16201Џ
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16212Е
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_16986dense_1_16988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_16233џ
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16244Е
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_16992dense_2_16994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_16257И
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_16968* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16972*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16974*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16980* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16982*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16986*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16988*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€№
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/StatefulPartitionedCall6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
„
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_16244

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®N
ѓ
E__inference_sequential_layer_call_and_return_conditional_losses_16292

inputs#
embedding_15906:
—Ж

lstm_16167:@

lstm_16169:@

lstm_16171:@
dense_16202:
†А
dense_16204:	А 
dense_1_16234:	А
dense_1_16236:
dense_2_16258:
dense_2_16260:
identityИҐdense/StatefulPartitionedCallҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/StatefulPartitionedCallҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/StatefulPartitionedCallҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpв
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_15906*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_15905Ё
dropout/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_15914Й
lstm/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
lstm_16167
lstm_16169
lstm_16171*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16166’
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_16180ь
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16202dense_16204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_16201Џ
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16212Е
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_16234dense_1_16236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_16233џ
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16244Е
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_16258dense_2_16260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_16257И
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_15906* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16167*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_16169*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16202* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16204*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16234*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_16236*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€№
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/StatefulPartitionedCall6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
а
п
)__inference_lstm_cell_layer_call_fn_19558

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
џ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_16212

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
РК
Ј
?__inference_lstm_layer_call_and_return_conditional_losses_18950

inputs9
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18808*
condR
while_cond_18807*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_19348

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
щR
Ґ
E__inference_sequential_layer_call_and_return_conditional_losses_17087
embedding_input#
embedding_17029:
—Ж

lstm_17033:@

lstm_17035:@

lstm_17037:@
dense_17041:
†А
dense_17043:	А 
dense_1_17047:	А
dense_1_17049:
dense_2_17053:
dense_2_17055:
identityИҐdense/StatefulPartitionedCallҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/StatefulPartitionedCallҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/StatefulPartitionedCallҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpл
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_17029*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_15905н
dropout/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_16818С
lstm/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
lstm_17033
lstm_17035
lstm_17037*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16789’
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_16180ь
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_17041dense_17043*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_16201М
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16378Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_17047dense_1_17049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_16233П
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16345Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_17053dense_2_17055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_16257И
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_17029* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_17033*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp
lstm_17035*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_17041* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_17043*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_17047*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_17049*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€∆
NoOpNoOp^dense/StatefulPartitionedCall-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/StatefulPartitionedCall6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
Ђq
щ
while_body_16024
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ґќ
®

lstm_while_body_17712&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0F
4lstm_while_lstm_cell_split_readvariableop_resource_0:@D
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:@@
.lstm_while_lstm_cell_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorD
2lstm_while_lstm_cell_split_readvariableop_resource:@B
4lstm_while_lstm_cell_split_1_readvariableop_resource:@>
,lstm_while_lstm_cell_readvariableop_resource:@ИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpН
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   њ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Й
$lstm/while/lstm_cell/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:i
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
"lstm/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?ѓ
 lstm/while/lstm_cell/dropout/MulMul'lstm/while/lstm_cell/ones_like:output:0+lstm/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
"lstm/while/lstm_cell/dropout/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ґ
9lstm/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform+lstm/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0p
+lstm/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>е
)lstm/while/lstm_cell/dropout/GreaterEqualGreaterEqualBlstm/while/lstm_cell/dropout/random_uniform/RandomUniform:output:04lstm/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    з
%lstm/while/lstm_cell/dropout/SelectV2SelectV2-lstm/while/lstm_cell/dropout/GreaterEqual:z:0$lstm/while/lstm_cell/dropout/Mul:z:0-lstm/while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?≥
"lstm/while/lstm_cell/dropout_1/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€{
$lstm/while/lstm_cell/dropout_1/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_1/SelectV2SelectV2/lstm/while/lstm_cell/dropout_1/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_1/Mul:z:0/lstm/while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?≥
"lstm/while/lstm_cell/dropout_2/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€{
$lstm/while/lstm_cell/dropout_2/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_2/SelectV2SelectV2/lstm/while/lstm_cell/dropout_2/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_2/Mul:z:0/lstm/while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?≥
"lstm/while/lstm_cell/dropout_3/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€{
$lstm/while/lstm_cell/dropout_3/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_3/SelectV2SelectV2/lstm/while/lstm_cell/dropout_3/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_3/Mul:z:0/lstm/while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
&lstm/while/lstm_cell/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:k
&lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Љ
 lstm/while/lstm_cell/ones_like_1Fill/lstm/while/lstm_cell/ones_like_1/Shape:output:0/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?µ
"lstm/while/lstm_cell/dropout_4/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
$lstm/while/lstm_cell/dropout_4/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_4/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_4/SelectV2SelectV2/lstm/while/lstm_cell/dropout_4/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_4/Mul:z:0/lstm/while/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?µ
"lstm/while/lstm_cell/dropout_5/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
$lstm/while/lstm_cell/dropout_5/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_5/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_5/SelectV2SelectV2/lstm/while/lstm_cell/dropout_5/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_5/Mul:z:0/lstm/while/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?µ
"lstm/while/lstm_cell/dropout_6/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
$lstm/while/lstm_cell/dropout_6/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_6/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_6/SelectV2SelectV2/lstm/while/lstm_cell/dropout_6/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_6/Mul:z:0/lstm/while/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€i
$lstm/while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?µ
"lstm/while/lstm_cell/dropout_7/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
$lstm/while/lstm_cell/dropout_7/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ї
;lstm/while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0r
-lstm/while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>л
+lstm/while/lstm_cell/dropout_7/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
&lstm/while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    п
'lstm/while/lstm_cell/dropout_7/SelectV2SelectV2/lstm/while/lstm_cell/dropout_7/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_7/Mul:z:0/lstm/while/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Є
lstm/while/lstm_cell/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Љ
lstm/while/lstm_cell/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Љ
lstm/while/lstm_cell/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Љ
lstm/while/lstm_cell/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ё
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЪ
lstm/while/lstm_cell/MatMulMatMullstm/while/lstm_cell/mul:z:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_1MatMullstm/while/lstm_cell/mul_1:z:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_2MatMullstm/while/lstm_cell/mul_2:z:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm/while/lstm_cell/MatMul_3MatMullstm/while/lstm_cell/mul_3:z:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ю
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0”
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitІ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ђ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Я
lstm/while/lstm_cell/mul_4Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Я
lstm/while/lstm_cell/mul_5Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Я
lstm/while/lstm_cell/mul_6Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Я
lstm/while/lstm_cell/mul_7Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Т
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¶
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul_4:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€£
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_5:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€С
lstm/while/lstm_cell/mul_8Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_6:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
lstm/while/lstm_cell/mul_9Mul lstm/while/lstm_cell/Sigmoid:y:0lstm/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_8:z:0lstm/while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_7:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€І
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€u
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/while/lstm_cell/mul_10Mul"lstm/while/lstm_cell/Sigmoid_2:y:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€„
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: Х
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: Ж
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_10:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€…
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
…
џ
D__inference_embedding_layer_call_and_return_conditional_losses_17998

inputs*
embedding_lookup_17988:
—Ж
identityИҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2є
embedding_lookupResourceGatherembedding_lookup_17988Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/17988*+
_output_shapes
:€€€€€€€€€2*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/17988*+
_output_shapes
:€€€€€€€€€2Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2П
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_17988* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2Т
NoOpNoOp7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€2: 2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
п
b
)__inference_dropout_2_layer_call_fn_19441

inputs
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЗФ
©
 sequential_lstm_while_body_15179<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0Q
?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0:@O
Asequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0:@K
9sequential_lstm_while_lstm_cell_readvariableop_resource_0:@"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorO
=sequential_lstm_while_lstm_cell_split_readvariableop_resource:@M
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resource:@I
7sequential_lstm_while_lstm_cell_readvariableop_resource:@ИҐ.sequential/lstm/while/lstm_cell/ReadVariableOpҐ0sequential/lstm/while/lstm_cell/ReadVariableOp_1Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_2Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_3Ґ4sequential/lstm/while/lstm_cell/split/ReadVariableOpҐ6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpШ
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ц
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
/sequential/lstm/while/lstm_cell/ones_like/ShapeShape@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:t
/sequential/lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?„
)sequential/lstm/while/lstm_cell/ones_likeFill8sequential/lstm/while/lstm_cell/ones_like/Shape:output:08sequential/lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
1sequential/lstm/while/lstm_cell/ones_like_1/ShapeShape#sequential_lstm_while_placeholder_2*
T0*
_output_shapes
:v
1sequential/lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ё
+sequential/lstm/while/lstm_cell/ones_like_1Fill:sequential/lstm/while/lstm_cell/ones_like_1/Shape:output:0:sequential/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€“
#sequential/lstm/while/lstm_cell/mulMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€‘
%sequential/lstm/while/lstm_cell/mul_1Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€‘
%sequential/lstm/while/lstm_cell/mul_2Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€‘
%sequential/lstm/while/lstm_cell/mul_3Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:02sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
/sequential/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :і
4sequential/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ю
%sequential/lstm/while/lstm_cell/splitSplit8sequential/lstm/while/lstm_cell/split/split_dim:output:0<sequential/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitї
&sequential/lstm/while/lstm_cell/MatMulMatMul'sequential/lstm/while/lstm_cell/mul:z:0.sequential/lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€њ
(sequential/lstm/while/lstm_cell/MatMul_1MatMul)sequential/lstm/while/lstm_cell/mul_1:z:0.sequential/lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€њ
(sequential/lstm/while/lstm_cell/MatMul_2MatMul)sequential/lstm/while/lstm_cell/mul_2:z:0.sequential/lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€њ
(sequential/lstm/while/lstm_cell/MatMul_3MatMul)sequential/lstm/while/lstm_cell/mul_3:z:0.sequential/lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€s
1sequential/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : і
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ф
'sequential/lstm/while/lstm_cell/split_1Split:sequential/lstm/while/lstm_cell/split_1/split_dim:output:0>sequential/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split»
'sequential/lstm/while/lstm_cell/BiasAddBiasAdd0sequential/lstm/while/lstm_cell/MatMul:product:00sequential/lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ћ
)sequential/lstm/while/lstm_cell/BiasAdd_1BiasAdd2sequential/lstm/while/lstm_cell/MatMul_1:product:00sequential/lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€ћ
)sequential/lstm/while/lstm_cell/BiasAdd_2BiasAdd2sequential/lstm/while/lstm_cell/MatMul_2:product:00sequential/lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€ћ
)sequential/lstm/while/lstm_cell/BiasAdd_3BiasAdd2sequential/lstm/while/lstm_cell/MatMul_3:product:00sequential/lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€є
%sequential/lstm/while/lstm_cell/mul_4Mul#sequential_lstm_while_placeholder_24sequential/lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€є
%sequential/lstm/while/lstm_cell/mul_5Mul#sequential_lstm_while_placeholder_24sequential/lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€є
%sequential/lstm/while/lstm_cell/mul_6Mul#sequential_lstm_while_placeholder_24sequential/lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€є
%sequential/lstm/while/lstm_cell/mul_7Mul#sequential_lstm_while_placeholder_24sequential/lstm/while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
.sequential/lstm/while/lstm_cell/ReadVariableOpReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0Д
3sequential/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ж
5sequential/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ж
5sequential/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Л
-sequential/lstm/while/lstm_cell/strided_sliceStridedSlice6sequential/lstm/while/lstm_cell/ReadVariableOp:value:0<sequential/lstm/while/lstm_cell/strided_slice/stack:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_1:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask«
(sequential/lstm/while/lstm_cell/MatMul_4MatMul)sequential/lstm/while/lstm_cell/mul_4:z:06sequential/lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
#sequential/lstm/while/lstm_cell/addAddV20sequential/lstm/while/lstm_cell/BiasAdd:output:02sequential/lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€Н
'sequential/lstm/while/lstm_cell/SigmoidSigmoid'sequential/lstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€™
0sequential/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       И
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_1StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_1:value:0>sequential/lstm/while/lstm_cell/strided_slice_1/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_5MatMul)sequential/lstm/while/lstm_cell/mul_5:z:08sequential/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€»
%sequential/lstm/while/lstm_cell/add_1AddV22sequential/lstm/while/lstm_cell/BiasAdd_1:output:02sequential/lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)sequential/lstm/while/lstm_cell/Sigmoid_1Sigmoid)sequential/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€≤
%sequential/lstm/while/lstm_cell/mul_8Mul-sequential/lstm/while/lstm_cell/Sigmoid_1:y:0#sequential_lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€™
0sequential/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   И
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_2StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_2:value:0>sequential/lstm/while/lstm_cell/strided_slice_2/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_6MatMul)sequential/lstm/while/lstm_cell/mul_6:z:08sequential/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€»
%sequential/lstm/while/lstm_cell/add_2AddV22sequential/lstm/while/lstm_cell/BiasAdd_2:output:02sequential/lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€Й
$sequential/lstm/while/lstm_cell/TanhTanh)sequential/lstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€µ
%sequential/lstm/while/lstm_cell/mul_9Mul+sequential/lstm/while/lstm_cell/Sigmoid:y:0(sequential/lstm/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
%sequential/lstm/while/lstm_cell/add_3AddV2)sequential/lstm/while/lstm_cell/mul_8:z:0)sequential/lstm/while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€™
0sequential/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   И
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_3StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_3:value:0>sequential/lstm/while/lstm_cell/strided_slice_3/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_7MatMul)sequential/lstm/while/lstm_cell/mul_7:z:08sequential/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€»
%sequential/lstm/while/lstm_cell/add_4AddV22sequential/lstm/while/lstm_cell/BiasAdd_3:output:02sequential/lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)sequential/lstm/while/lstm_cell/Sigmoid_2Sigmoid)sequential/lstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€Л
&sequential/lstm/while/lstm_cell/Tanh_1Tanh)sequential/lstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
&sequential/lstm/while/lstm_cell/mul_10Mul-sequential/lstm/while/lstm_cell/Sigmoid_2:y:0*sequential/lstm/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Г
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder*sequential/lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“]
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :М
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :І
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: Й
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ™
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations^sequential/lstm/while/NoOp*
T0*
_output_shapes
: Й
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ґ
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: І
 sequential/lstm/while/Identity_4Identity*sequential/lstm/while/lstm_cell/mul_10:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶
 sequential/lstm/while/Identity_5Identity)sequential/lstm/while/lstm_cell/add_3:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ц
sequential/lstm/while/NoOpNoOp/^sequential/lstm/while/lstm_cell/ReadVariableOp1^sequential/lstm/while/lstm_cell/ReadVariableOp_11^sequential/lstm/while/lstm_cell/ReadVariableOp_21^sequential/lstm/while/lstm_cell/ReadVariableOp_35^sequential/lstm/while/lstm_cell/split/ReadVariableOp7^sequential/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"t
7sequential_lstm_while_lstm_cell_readvariableop_resource9sequential_lstm_while_lstm_cell_readvariableop_resource_0"Д
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resourceAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0"А
=sequential_lstm_while_lstm_cell_split_readvariableop_resource?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"и
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2`
.sequential/lstm/while/lstm_cell/ReadVariableOp.sequential/lstm/while/lstm_cell/ReadVariableOp2d
0sequential/lstm/while/lstm_cell/ReadVariableOp_10sequential/lstm/while/lstm_cell/ReadVariableOp_12d
0sequential/lstm/while/lstm_cell/ReadVariableOp_20sequential/lstm/while/lstm_cell/ReadVariableOp_22d
0sequential/lstm/while/lstm_cell/ReadVariableOp_30sequential/lstm/while/lstm_cell/ReadVariableOp_32l
4sequential/lstm/while/lstm_cell/split/ReadVariableOp4sequential/lstm/while/lstm_cell/split/ReadVariableOp2p
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ЁШ
√
!__inference__traced_restore_20068
file_prefix9
%assignvariableop_embedding_embeddings:
—Ж3
assignvariableop_1_dense_kernel:
†А,
assignvariableop_2_dense_bias:	А4
!assignvariableop_3_dense_1_kernel:	А-
assignvariableop_4_dense_1_bias:3
!assignvariableop_5_dense_2_kernel:-
assignvariableop_6_dense_2_bias::
(assignvariableop_7_lstm_lstm_cell_kernel:@D
2assignvariableop_8_lstm_lstm_cell_recurrent_kernel:@4
&assignvariableop_9_lstm_lstm_cell_bias:@'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: C
/assignvariableop_12_adam_m_embedding_embeddings:
—ЖC
/assignvariableop_13_adam_v_embedding_embeddings:
—ЖB
0assignvariableop_14_adam_m_lstm_lstm_cell_kernel:@B
0assignvariableop_15_adam_v_lstm_lstm_cell_kernel:@L
:assignvariableop_16_adam_m_lstm_lstm_cell_recurrent_kernel:@L
:assignvariableop_17_adam_v_lstm_lstm_cell_recurrent_kernel:@<
.assignvariableop_18_adam_m_lstm_lstm_cell_bias:@<
.assignvariableop_19_adam_v_lstm_lstm_cell_bias:@;
'assignvariableop_20_adam_m_dense_kernel:
†А;
'assignvariableop_21_adam_v_dense_kernel:
†А4
%assignvariableop_22_adam_m_dense_bias:	А4
%assignvariableop_23_adam_v_dense_bias:	А<
)assignvariableop_24_adam_m_dense_1_kernel:	А<
)assignvariableop_25_adam_v_dense_1_kernel:	А5
'assignvariableop_26_adam_m_dense_1_bias:5
'assignvariableop_27_adam_v_dense_1_bias:;
)assignvariableop_28_adam_m_dense_2_kernel:;
)assignvariableop_29_adam_v_dense_2_kernel:5
'assignvariableop_30_adam_m_dense_2_bias:5
'assignvariableop_31_adam_v_dense_2_bias:%
assignvariableop_32_total_1: %
assignvariableop_33_count_1: #
assignvariableop_34_total: #
assignvariableop_35_count: 
identity_37ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9—
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*ч
valueнBк%B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Џ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*™
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_7AssignVariableOp(assignvariableop_7_lstm_lstm_cell_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_8AssignVariableOp2assignvariableop_8_lstm_lstm_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_9AssignVariableOp&assignvariableop_9_lstm_lstm_cell_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_12AssignVariableOp/assignvariableop_12_adam_m_embedding_embeddingsIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_13AssignVariableOp/assignvariableop_13_adam_v_embedding_embeddingsIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_m_lstm_lstm_cell_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_v_lstm_lstm_cell_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_16AssignVariableOp:assignvariableop_16_adam_m_lstm_lstm_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_17AssignVariableOp:assignvariableop_17_adam_v_lstm_lstm_cell_recurrent_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_m_lstm_lstm_cell_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_v_lstm_lstm_cell_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_m_dense_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_v_dense_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_m_dense_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_v_dense_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_dense_1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_dense_1_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_m_dense_1_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_v_dense_1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_dense_2_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_dense_2_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_m_dense_2_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_v_dense_2_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 з
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: ‘
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
E
)__inference_dropout_1_layer_call_fn_19381

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16212a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ю

щ
*__inference_sequential_layer_call_fn_17197

inputs
unknown:
—Ж
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
†А
	unknown_4:	А
	unknown_5:	А
	unknown_6:
	unknown_7:
	unknown_8:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_16292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
у
b
)__inference_dropout_1_layer_call_fn_19386

inputs
identityИҐStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_16378p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
„
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_19446

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞
Њ
while_cond_15476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_15476___redundant_placeholder03
/while_while_cond_15476___redundant_placeholder13
/while_while_cond_15476___redundant_placeholder23
/while_while_cond_15476___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
ѕМ
Ц
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19819

inputs
states_0
states_1/
split_readvariableop_resource:@-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
mulMulinputsdropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ю
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:€€€€€€€€€S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
mul_4Mulstates_0dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
mul_5Mulstates_0dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
mul_6Mulstates_0dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
mul_7Mulstates_0dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: С
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
Љ	
Ґ
lstm_while_cond_17711&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_17711___redundant_placeholder0=
9lstm_while_lstm_while_cond_17711___redundant_placeholder1=
9lstm_while_lstm_while_cond_17711___redundant_placeholder2=
9lstm_while_lstm_while_cond_17711___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ѕ
Х
%__inference_dense_layer_call_fn_19357

inputs
unknown:
†А
	unknown_0:	А
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_16201p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
“
£
__inference_loss_fn_2_19505D
5dense_bias_regularizer_l2loss_readvariableop_resource:	А
identityИҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpЯ
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: \
IdentityIdentitydense/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^dense/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp
З"
Ѕ
while_body_15798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_lstm_cell_15822_0:@%
while_lstm_cell_15824_0:@)
while_lstm_cell_15826_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_lstm_cell_15822:@#
while_lstm_cell_15824:@'
while_lstm_cell_15826:@ИҐ'while/lstm_cell/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0†
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15822_0while_lstm_cell_15824_0while_lstm_cell_15826_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15739ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Н
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Н
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_15822while_lstm_cell_15822_0"0
while_lstm_cell_15824while_lstm_cell_15824_0"0
while_lstm_cell_15826while_lstm_cell_15826_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
£
C
'__inference_flatten_layer_call_fn_19342

inputs
identityЃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_16180a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
©
C
'__inference_dropout_layer_call_fn_18003

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_15914d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ђq
щ
while_body_18178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
в
Ў
B__inference_dense_1_layer_call_and_return_conditional_losses_16233

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€џ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Њ
Ф
'__inference_dense_2_layer_call_fn_19467

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_16257o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
в
Ў
B__inference_dense_1_layer_call_and_return_conditional_losses_19431

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€џ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∞
Њ
while_cond_15797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_15797___redundant_placeholder03
/while_while_cond_15797___redundant_placeholder13
/while_while_cond_15797___redundant_placeholder23
/while_while_cond_15797___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
е
`
B__inference_dropout_layer_call_and_return_conditional_losses_15914

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:€€€€€€€€€2_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
С

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_16378

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
ђ
__inference_loss_fn_1_19496K
7dense_kernel_regularizer_l2loss_readvariableop_resource:
†А
identityИҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOp®
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp
Ђq
щ
while_body_18808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0A
/while_lstm_cell_split_readvariableop_resource_0:@?
1while_lstm_cell_split_1_readvariableop_resource_0:@;
)while_lstm_cell_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor?
-while_lstm_cell_split_readvariableop_resource:@=
/while_lstm_cell_split_1_readvariableop_resource:@9
'while_lstm_cell_readvariableop_resource:@ИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:f
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ф
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype0ќ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitЛ
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€П
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ф
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЧ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€√
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_16180

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
К

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_19458

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…–
Ј
?__inference_lstm_layer_call_and_return_conditional_losses_19329

inputs9
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:†
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ї
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Е
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19123*
condR
while_cond_19122*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
§

a
B__inference_dropout_layer_call_and_return_conditional_losses_18025

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>™
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€2e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
С

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_19403

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
…
џ
D__inference_embedding_layer_call_and_return_conditional_losses_15905

inputs*
embedding_lookup_15895:
—Ж
identityИҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2є
embedding_lookupResourceGatherembedding_lookup_15895Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/15895*+
_output_shapes
:€€€€€€€€€2*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/15895*+
_output_shapes
:€€€€€€€€€2Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2П
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_15895* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2Т
NoOpNoOp7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€2: 2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Э
E
)__inference_dropout_2_layer_call_fn_19436

inputs
identityѓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_16244`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
§

a
B__inference_dropout_layer_call_and_return_conditional_losses_16818

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>™
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€2e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:€€€€€€€€€2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
є

В
*__inference_sequential_layer_call_fn_16315
embedding_input
unknown:
—Ж
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:
†А
	unknown_4:	А
	unknown_5:	А
	unknown_6:
	unknown_7:
	unknown_8:
identityИҐStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_16292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€2
)
_user_specified_nameembedding_input
∞
Њ
while_cond_18177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18177___redundant_placeholder03
/while_while_cond_18177___redundant_placeholder13
/while_while_cond_18177___redundant_placeholder23
/while_while_cond_18177___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
њМ
Ф
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15739

inputs

states
states_1/
split_readvariableop_resource:@-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Р
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:Р
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ђ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
mulMulinputsdropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ю
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:€€€€€€€€€S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€c
mul_4Mulstatesdropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_5Mulstatesdropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_6Mulstatesdropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_7Mulstatesdropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: С
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
љ
‘
@__inference_dense_layer_call_and_return_conditional_losses_19376

inputs2
matmul_readvariableop_resource:
†А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АП
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А„
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
£

)__inference_embedding_layer_call_fn_17984

inputs
unknown:
—Ж
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_15905s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€2: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ф
Ѓ
$__inference_lstm_layer_call_fn_18069

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16789s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
∞
Њ
while_cond_19122
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_19122___redundant_placeholder03
/while_while_cond_19122___redundant_placeholder13
/while_while_cond_19122___redundant_placeholder23
/while_while_cond_19122___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
яP
Ф
D__inference_lstm_cell_layer_call_and_return_conditional_losses_15463

inputs

states
states_1/
split_readvariableop_resource:@-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ю
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:€€€€€€€€€S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€\
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€Х
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: С
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
∞
Њ
while_cond_18492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18492___redundant_placeholder03
/while_while_cond_18492___redundant_placeholder13
/while_while_cond_18492___redundant_placeholder23
/while_while_cond_18492___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
К

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_16345

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы
`
'__inference_dropout_layer_call_fn_18008

inputs
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_16818s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€222
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ъЄ
—
E__inference_sequential_layer_call_and_return_conditional_losses_17977

inputs4
 embedding_embedding_lookup_17529:
—Ж>
,lstm_lstm_cell_split_readvariableop_resource:@<
.lstm_lstm_cell_split_1_readvariableop_resource:@8
&lstm_lstm_cell_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
†А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐ,dense/bias/Regularizer/L2Loss/ReadVariableOpҐ.dense/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ.dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐembedding/embedding_lookupҐ6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/while_
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2б
embedding/embedding_lookupResourceGather embedding_embedding_lookup_17529embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/17529*+
_output_shapes
:€€€€€€€€€2*
dtype0њ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/17529*+
_output_shapes
:€€€€€€€€€2Х
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€2Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?†
dropout/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€2s
dropout/dropout/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:†
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>¬
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ј
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€2[

lstm/ShapeShape!dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Т
lstm/transpose	Transpose!dropout/dropout/SelectV2:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€√
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Л
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   п
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskk
lstm/lstm_cell/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:c
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?§
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Э
lstm/lstm_cell/dropout/MulMul!lstm/lstm_cell/ones_like:output:0%lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€m
lstm/lstm_cell/dropout/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:™
3lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform%lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0j
%lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>”
#lstm/lstm_cell/dropout/GreaterEqualGreaterEqual<lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:0.lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ѕ
lstm/lstm_cell/dropout/SelectV2SelectV2'lstm/lstm_cell/dropout/GreaterEqual:z:0lstm/lstm_cell/dropout/Mul:z:0'lstm/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?°
lstm/lstm_cell/dropout_1/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/dropout_1/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_1/SelectV2SelectV2)lstm/lstm_cell/dropout_1/GreaterEqual:z:0 lstm/lstm_cell/dropout_1/Mul:z:0)lstm/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?°
lstm/lstm_cell/dropout_2/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/dropout_2/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_2/SelectV2SelectV2)lstm/lstm_cell/dropout_2/GreaterEqual:z:0 lstm/lstm_cell/dropout_2/Mul:z:0)lstm/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?°
lstm/lstm_cell/dropout_3/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/dropout_3/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_3/SelectV2SelectV2)lstm/lstm_cell/dropout_3/GreaterEqual:z:0 lstm/lstm_cell/dropout_3/Mul:z:0)lstm/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
 lstm/lstm_cell/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:e
 lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?™
lstm/lstm_cell/ones_like_1Fill)lstm/lstm_cell/ones_like_1/Shape:output:0)lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?£
lstm/lstm_cell/dropout_4/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
lstm/lstm_cell/dropout_4/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_4/SelectV2SelectV2)lstm/lstm_cell/dropout_4/GreaterEqual:z:0 lstm/lstm_cell/dropout_4/Mul:z:0)lstm/lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?£
lstm/lstm_cell/dropout_5/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
lstm/lstm_cell/dropout_5/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_5/SelectV2SelectV2)lstm/lstm_cell/dropout_5/GreaterEqual:z:0 lstm/lstm_cell/dropout_5/Mul:z:0)lstm/lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?£
lstm/lstm_cell/dropout_6/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
lstm/lstm_cell/dropout_6/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_6/SelectV2SelectV2)lstm/lstm_cell/dropout_6/GreaterEqual:z:0 lstm/lstm_cell/dropout_6/Mul:z:0)lstm/lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?£
lstm/lstm_cell/dropout_7/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
lstm/lstm_cell/dropout_7/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:Ѓ
5lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0l
'lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ў
%lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
 lstm/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    „
!lstm/lstm_cell/dropout_7/SelectV2SelectV2)lstm/lstm_cell/dropout_7/GreaterEqual:z:0 lstm/lstm_cell/dropout_7/Mul:z:0)lstm/lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
lstm/lstm_cell/mulMullstm/strided_slice_2:output:0(lstm/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/lstm_cell/mul_1Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/lstm_cell/mul_2Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
lstm/lstm_cell/mul_3Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Р
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitИ
lstm/lstm_cell/MatMulMatMullstm/lstm_cell/mul:z:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_1MatMullstm/lstm_cell/mul_1:z:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_2MatMullstm/lstm_cell/mul_2:z:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€М
lstm/lstm_cell/MatMul_3MatMullstm/lstm_cell/mul_3:z:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Р
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ѕ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitХ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€Щ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€О
lstm/lstm_cell/mul_4Mullstm/zeros:output:0*lstm/lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
lstm/lstm_cell/mul_5Mullstm/zeros:output:0*lstm/lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
lstm/lstm_cell/mul_6Mullstm/zeros:output:0*lstm/lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
lstm/lstm_cell/mul_7Mullstm/zeros:output:0*lstm/lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ґ
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskФ
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul_4:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€С
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_5:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm/lstm_cell/mul_8Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_6:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm/lstm_cell/mul_9Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€Г
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_8:z:0lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_7:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€o
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€i
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€З
lstm/lstm_cell/mul_10Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   «
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ≥

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_17712*!
condR
lstm_while_cond_17711*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Ж
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   —
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:†
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          •
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   {
flatten/ReshapeReshapelstm/transpose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†В
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?Л
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А_
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:°
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>≈
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Љ
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_1/MatMulMatMul#dropout_1/dropout/SelectV2:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?М
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
dropout_2/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:†
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>ƒ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ї
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ц
dense_2/MatMulMatMul#dropout_2/dropout/SelectV2:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOp embedding_embedding_lookup_17529* 
_output_shapes
:
—Ж*
dtype0Т
'embedding/embeddings/Regularizer/L2LossL2Loss>embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:00embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: §
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: †
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
†А*
dtype0В
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: П
,dense/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
dense/bias/Regularizer/L2LossL2Loss4dense/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;С
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0&dense/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ш
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ж
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Э
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
dense_1/bias/Regularizer/L2LossL2Loss6dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0(dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/L2Loss/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp/^dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding/embedding_lookup7^embedding/embeddings/Regularizer/L2Loss/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_36^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€2: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,dense/bias/Regularizer/L2Loss/ReadVariableOp,dense/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/L2Loss/ReadVariableOp.dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2p
6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp6embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ф
Ѓ
$__inference_lstm_layer_call_fn_18058

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_16166s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
џ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_19391

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
√I
Є
__inference__traced_save_19950
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop:
6savev2_adam_m_embedding_embeddings_read_readvariableop:
6savev2_adam_v_embedding_embeddings_read_readvariableop;
7savev2_adam_m_lstm_lstm_cell_kernel_read_readvariableop;
7savev2_adam_v_lstm_lstm_cell_kernel_read_readvariableopE
Asavev2_adam_m_lstm_lstm_cell_recurrent_kernel_read_readvariableopE
Asavev2_adam_v_lstm_lstm_cell_recurrent_kernel_read_readvariableop9
5savev2_adam_m_lstm_lstm_cell_bias_read_readvariableop9
5savev2_adam_v_lstm_lstm_cell_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableop4
0savev2_adam_m_dense_2_kernel_read_readvariableop4
0savev2_adam_v_dense_2_kernel_read_readvariableop2
.savev2_adam_m_dense_2_bias_read_readvariableop2
.savev2_adam_v_dense_2_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ќ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*ч
valueнBк%B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B є
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop6savev2_adam_m_embedding_embeddings_read_readvariableop6savev2_adam_v_embedding_embeddings_read_readvariableop7savev2_adam_m_lstm_lstm_cell_kernel_read_readvariableop7savev2_adam_v_lstm_lstm_cell_kernel_read_readvariableopAsavev2_adam_m_lstm_lstm_cell_recurrent_kernel_read_readvariableopAsavev2_adam_v_lstm_lstm_cell_recurrent_kernel_read_readvariableop5savev2_adam_m_lstm_lstm_cell_bias_read_readvariableop5savev2_adam_v_lstm_lstm_cell_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*≥
_input_shapes°
Ю: :
—Ж:
†А:А:	А::::@:@:@: : :
—Ж:
—Ж:@:@:@:@:@:@:
†А:
†А:А:А:	А:	А::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
—Ж:&"
 
_output_shapes
:
†А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@:$	 

_output_shapes

:@: 


_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
—Ж:&"
 
_output_shapes
:
—Ж:$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
†А:&"
 
_output_shapes
:
†А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А:%!

_output_shapes
:	А: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
ќК
є
?__inference_lstm_layer_call_and_return_conditional_losses_18320
inputs_09
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_18178*
condR
while_cond_18177*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
…–
Ј
?__inference_lstm_layer_call_and_return_conditional_losses_16789

inputs9
'lstm_cell_split_readvariableop_resource:@7
)lstm_cell_split_1_readvariableop_resource:@3
!lstm_cell_readvariableop_resource:@
identityИҐ5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpҐ7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maska
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:†
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ї
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:§
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL> 
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    √
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Е
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Й
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Љ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splity
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€}
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ж
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЕ
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€]
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€s
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€t
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€_
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_16583*
condR
while_cond_16582*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2€€€€€€€€€*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Я
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
(lstm/lstm_cell/kernel/Regularizer/L2LossL2Loss?lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
%lstm/lstm_cell/kernel/Regularizer/mulMul0lstm/lstm_cell/kernel/Regularizer/mul/x:output:01lstm/lstm_cell/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0Р
&lstm/lstm_cell/bias/Regularizer/L2LossL2Loss=lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%lstm/lstm_cell/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ђ
#lstm/lstm_cell/bias/Regularizer/mulMul.lstm/lstm_cell/bias/Regularizer/mul/x:output:0/lstm/lstm_cell/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2ц
NoOpNoOp6^lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp8^lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€2: : : 2n
5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp5lstm/lstm_cell/bias/Regularizer/L2Loss/ReadVariableOp2r
7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp7lstm/lstm_cell/kernel/Regularizer/L2Loss/ReadVariableOp24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ї
serving_default¶
K
embedding_input8
!serving_default_embedding_input:0€€€€€€€€€2;
dense_20
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:µЋ
Ё
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
µ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Љ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator"
_tf_keras_layer
Џ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec"
_tf_keras_rnn_layer
•
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
Љ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator"
_tf_keras_layer
ї
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
Љ
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator"
_tf_keras_layer
ї
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias"
_tf_keras_layer
f
0
V1
W2
X3
64
75
E6
F7
T8
U9"
trackable_list_wrapper
f
0
V1
W2
X3
64
75
E6
F7
T8
U9"
trackable_list_wrapper
C
Y0
Z1
[2
\3
]4"
trackable_list_wrapper
 
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ё
ctrace_0
dtrace_1
etrace_2
ftrace_32т
*__inference_sequential_layer_call_fn_16315
*__inference_sequential_layer_call_fn_17197
*__inference_sequential_layer_call_fn_17222
*__inference_sequential_layer_call_fn_16965њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zctrace_0zdtrace_1zetrace_2zftrace_3
…
gtrace_0
htrace_1
itrace_2
jtrace_32ё
E__inference_sequential_layer_call_and_return_conditional_losses_17525
E__inference_sequential_layer_call_and_return_conditional_losses_17977
E__inference_sequential_layer_call_and_return_conditional_losses_17026
E__inference_sequential_layer_call_and_return_conditional_losses_17087њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zgtrace_0zhtrace_1zitrace_2zjtrace_3
”B–
 __inference__wrapped_model_15338embedding_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
k
_variables
l_iterations
m_learning_rate
n_index_dict
o
_momentums
p_velocities
q_update_step_xla"
experimentalOptimizer
,
rserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
≠
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
н
xtrace_02–
)__inference_embedding_layer_call_fn_17984Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zxtrace_0
И
ytrace_02л
D__inference_embedding_layer_call_and_return_conditional_losses_17998Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zytrace_0
(:&
—Ж2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ѕ
trace_0
Аtrace_12И
'__inference_dropout_layer_call_fn_18003
'__inference_dropout_layer_call_fn_18008≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0zАtrace_1
щ
Бtrace_0
Вtrace_12Њ
B__inference_dropout_layer_call_and_return_conditional_losses_18013
B__inference_dropout_layer_call_and_return_conditional_losses_18025≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zБtrace_0zВtrace_1
"
_generic_user_object
5
V0
W1
X2"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
њ
Еstates
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
в
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32п
$__inference_lstm_layer_call_fn_18036
$__inference_lstm_layer_call_fn_18047
$__inference_lstm_layer_call_fn_18058
$__inference_lstm_layer_call_fn_18069‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
ќ
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_32џ
?__inference_lstm_layer_call_and_return_conditional_losses_18320
?__inference_lstm_layer_call_and_return_conditional_losses_18699
?__inference_lstm_layer_call_and_return_conditional_losses_18950
?__inference_lstm_layer_call_and_return_conditional_losses_19329‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0zРtrace_1zСtrace_2zТtrace_3
"
_generic_user_object
А
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щ_random_generator
Ъ
state_size

Vkernel
Wrecurrent_kernel
Xbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
н
†trace_02ќ
'__inference_flatten_layer_call_fn_19342Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z†trace_0
И
°trace_02й
B__inference_flatten_layer_call_and_return_conditional_losses_19348Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z°trace_0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
≤
Ґnon_trainable_variables
£layers
§metrics
 •layer_regularization_losses
¶layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
л
Іtrace_02ћ
%__inference_dense_layer_call_fn_19357Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zІtrace_0
Ж
®trace_02з
@__inference_dense_layer_call_and_return_conditional_losses_19376Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z®trace_0
 :
†А2dense/kernel
:А2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
«
Ѓtrace_0
ѓtrace_12М
)__inference_dropout_1_layer_call_fn_19381
)__inference_dropout_1_layer_call_fn_19386≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЃtrace_0zѓtrace_1
э
∞trace_0
±trace_12¬
D__inference_dropout_1_layer_call_and_return_conditional_losses_19391
D__inference_dropout_1_layer_call_and_return_conditional_losses_19403≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0z±trace_1
"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
≤
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
н
Јtrace_02ќ
'__inference_dense_1_layer_call_fn_19412Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0
И
Єtrace_02й
B__inference_dense_1_layer_call_and_return_conditional_losses_19431Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЄtrace_0
!:	А2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
«
Њtrace_0
њtrace_12М
)__inference_dropout_2_layer_call_fn_19436
)__inference_dropout_2_layer_call_fn_19441≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЊtrace_0zњtrace_1
э
јtrace_0
Ѕtrace_12¬
D__inference_dropout_2_layer_call_and_return_conditional_losses_19446
D__inference_dropout_2_layer_call_and_return_conditional_losses_19458≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0zЅtrace_1
"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
н
«trace_02ќ
'__inference_dense_2_layer_call_fn_19467Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
И
»trace_02й
B__inference_dense_2_layer_call_and_return_conditional_losses_19478Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
 :2dense_2/kernel
:2dense_2/bias
':%@2lstm/lstm_cell/kernel
1:/@2lstm/lstm_cell/recurrent_kernel
!:@2lstm/lstm_cell/bias
ќ
…trace_02ѓ
__inference_loss_fn_0_19487П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z…trace_0
ќ
 trace_02ѓ
__inference_loss_fn_1_19496П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z trace_0
ќ
Ћtrace_02ѓ
__inference_loss_fn_2_19505П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЋtrace_0
ќ
ћtrace_02ѓ
__inference_loss_fn_3_19514П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zћtrace_0
ќ
Ќtrace_02ѓ
__inference_loss_fn_4_19523П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЌtrace_0
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
ќ0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
*__inference_sequential_layer_call_fn_16315embedding_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
*__inference_sequential_layer_call_fn_17197inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
*__inference_sequential_layer_call_fn_17222inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ДBБ
*__inference_sequential_layer_call_fn_16965embedding_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_17525inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_17977inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
E__inference_sequential_layer_call_and_return_conditional_losses_17026embedding_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
E__inference_sequential_layer_call_and_return_conditional_losses_17087embedding_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“
l0
–1
—2
“3
”4
‘5
’6
÷7
„8
Ў9
ў10
Џ11
џ12
№13
Ё14
ё15
я16
а17
б18
в19
г20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
–0
“1
‘2
÷3
Ў4
Џ5
№6
ё7
а8
в9"
trackable_list_wrapper
p
—0
”1
’2
„3
ў4
џ5
Ё6
я7
б8
г9"
trackable_list_wrapper
њ2Љє
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
“Bѕ
#__inference_signature_wrapper_17144embedding_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_embedding_layer_call_fn_17984inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_embedding_layer_call_and_return_conditional_losses_17998inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
'__inference_dropout_layer_call_fn_18003inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
'__inference_dropout_layer_call_fn_18008inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_18013inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_18025inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ
дtrace_02ѓ
__inference_loss_fn_5_19532П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zдtrace_0
ќ
еtrace_02ѓ
__inference_loss_fn_6_19541П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zеtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
$__inference_lstm_layer_call_fn_18036inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
$__inference_lstm_layer_call_fn_18047inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
$__inference_lstm_layer_call_fn_18058inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
$__inference_lstm_layer_call_fn_18069inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
?__inference_lstm_layer_call_and_return_conditional_losses_18320inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
?__inference_lstm_layer_call_and_return_conditional_losses_18699inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
•BҐ
?__inference_lstm_layer_call_and_return_conditional_losses_18950inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
•BҐ
?__inference_lstm_layer_call_and_return_conditional_losses_19329inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5
V0
W1
X2"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
—
лtrace_0
мtrace_12Ц
)__inference_lstm_cell_layer_call_fn_19558
)__inference_lstm_cell_layer_call_fn_19575љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zлtrace_0zмtrace_1
З
нtrace_0
оtrace_12ћ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19665
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19819љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zнtrace_0zоtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_flatten_layer_call_fn_19342inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_flatten_layer_call_and_return_conditional_losses_19348inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_dict_wrapper
ўB÷
%__inference_dense_layer_call_fn_19357inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
@__inference_dense_layer_call_and_return_conditional_losses_19376inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
)__inference_dropout_1_layer_call_fn_19381inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
)__inference_dropout_1_layer_call_fn_19386inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
D__inference_dropout_1_layer_call_and_return_conditional_losses_19391inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
D__inference_dropout_1_layer_call_and_return_conditional_losses_19403inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_dense_1_layer_call_fn_19412inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_dense_1_layer_call_and_return_conditional_losses_19431inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
)__inference_dropout_2_layer_call_fn_19436inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
)__inference_dropout_2_layer_call_fn_19441inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
D__inference_dropout_2_layer_call_and_return_conditional_losses_19446inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
D__inference_dropout_2_layer_call_and_return_conditional_losses_19458inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_dense_2_layer_call_fn_19467inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_dense_2_layer_call_and_return_conditional_losses_19478inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≤Bѓ
__inference_loss_fn_0_19487"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_1_19496"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_2_19505"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_3_19514"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_4_19523"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
R
п	variables
р	keras_api

сtotal

тcount"
_tf_keras_metric
c
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs"
_tf_keras_metric
-:+
—Ж2Adam/m/embedding/embeddings
-:+
—Ж2Adam/v/embedding/embeddings
,:*@2Adam/m/lstm/lstm_cell/kernel
,:*@2Adam/v/lstm/lstm_cell/kernel
6:4@2&Adam/m/lstm/lstm_cell/recurrent_kernel
6:4@2&Adam/v/lstm/lstm_cell/recurrent_kernel
&:$@2Adam/m/lstm/lstm_cell/bias
&:$@2Adam/v/lstm/lstm_cell/bias
%:#
†А2Adam/m/dense/kernel
%:#
†А2Adam/v/dense/kernel
:А2Adam/m/dense/bias
:А2Adam/v/dense/bias
&:$	А2Adam/m/dense_1/kernel
&:$	А2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
%:#2Adam/m/dense_2/kernel
%:#2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
≤Bѓ
__inference_loss_fn_5_19532"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_6_19541"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
)__inference_lstm_cell_layer_call_fn_19558inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
МBЙ
)__inference_lstm_cell_layer_call_fn_19575inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19665inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ІB§
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19819inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
с0
т1"
trackable_list_wrapper
.
п	variables"
_generic_user_object
:  (2total
:  (2count
0
х0
ц1"
trackable_list_wrapper
.
у	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЭ
 __inference__wrapped_model_15338y
VXW67EFTU8Ґ5
.Ґ+
)К&
embedding_input€€€€€€€€€2
™ "1™.
,
dense_2!К
dense_2€€€€€€€€€™
B__inference_dense_1_layer_call_and_return_conditional_losses_19431dEF0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Д
'__inference_dense_1_layer_call_fn_19412YEF0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "!К
unknown€€€€€€€€€©
B__inference_dense_2_layer_call_and_return_conditional_losses_19478cTU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Г
'__inference_dense_2_layer_call_fn_19467XTU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€©
@__inference_dense_layer_call_and_return_conditional_losses_19376e670Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Г
%__inference_dense_layer_call_fn_19357Z670Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ ""К
unknown€€€€€€€€€А≠
D__inference_dropout_1_layer_call_and_return_conditional_losses_19391e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ ≠
D__inference_dropout_1_layer_call_and_return_conditional_losses_19403e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ З
)__inference_dropout_1_layer_call_fn_19381Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ ""К
unknown€€€€€€€€€АЗ
)__inference_dropout_1_layer_call_fn_19386Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ ""К
unknown€€€€€€€€€АЂ
D__inference_dropout_2_layer_call_and_return_conditional_losses_19446c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ђ
D__inference_dropout_2_layer_call_and_return_conditional_losses_19458c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Е
)__inference_dropout_2_layer_call_fn_19436X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "!К
unknown€€€€€€€€€Е
)__inference_dropout_2_layer_call_fn_19441X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "!К
unknown€€€€€€€€€±
B__inference_dropout_layer_call_and_return_conditional_losses_18013k7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€2
p 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ ±
B__inference_dropout_layer_call_and_return_conditional_losses_18025k7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€2
p
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ Л
'__inference_dropout_layer_call_fn_18003`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€2
p 
™ "%К"
unknown€€€€€€€€€2Л
'__inference_dropout_layer_call_fn_18008`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€2
p
™ "%К"
unknown€€€€€€€€€2Ѓ
D__inference_embedding_layer_call_and_return_conditional_losses_17998f/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ И
)__inference_embedding_layer_call_fn_17984[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%К"
unknown€€€€€€€€€2™
B__inference_flatten_layer_call_and_return_conditional_losses_19348d3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€2
™ "-Ґ*
#К 
tensor_0€€€€€€€€€†
Ъ Д
'__inference_flatten_layer_call_fn_19342Y3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€2
™ ""К
unknown€€€€€€€€€†C
__inference_loss_fn_0_19487$Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_1_19496$6Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_2_19505$7Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_3_19514$EҐ

Ґ 
™ "К
unknown C
__inference_loss_fn_4_19523$FҐ

Ґ 
™ "К
unknown C
__inference_loss_fn_5_19532$VҐ

Ґ 
™ "К
unknown C
__inference_loss_fn_6_19541$XҐ

Ґ 
™ "К
unknown Ё
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19665ФVXWАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p 
™ "ЙҐЕ
~Ґ{
$К!

tensor_0_0€€€€€€€€€
SЪP
&К#
tensor_0_1_0€€€€€€€€€
&К#
tensor_0_1_1€€€€€€€€€
Ъ Ё
D__inference_lstm_cell_layer_call_and_return_conditional_losses_19819ФVXWАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p
™ "ЙҐЕ
~Ґ{
$К!

tensor_0_0€€€€€€€€€
SЪP
&К#
tensor_0_1_0€€€€€€€€€
&К#
tensor_0_1_1€€€€€€€€€
Ъ ∞
)__inference_lstm_cell_layer_call_fn_19558ВVXWАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p 
™ "xҐu
"К
tensor_0€€€€€€€€€
OЪL
$К!

tensor_1_0€€€€€€€€€
$К!

tensor_1_1€€€€€€€€€∞
)__inference_lstm_cell_layer_call_fn_19575ВVXWАҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p
™ "xҐu
"К
tensor_0€€€€€€€€€
OЪL
$К!

tensor_1_0€€€€€€€€€
$К!

tensor_1_1€€€€€€€€€’
?__inference_lstm_layer_call_and_return_conditional_losses_18320СVXWOҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ ’
?__inference_lstm_layer_call_and_return_conditional_losses_18699СVXWOҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ ї
?__inference_lstm_layer_call_and_return_conditional_losses_18950xVXW?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2

 
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ ї
?__inference_lstm_layer_call_and_return_conditional_losses_19329xVXW?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2

 
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€2
Ъ ѓ
$__inference_lstm_layer_call_fn_18036ЖVXWOҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ѓ
$__inference_lstm_layer_call_fn_18047ЖVXWOҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€Х
$__inference_lstm_layer_call_fn_18058mVXW?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2

 
p 

 
™ "%К"
unknown€€€€€€€€€2Х
$__inference_lstm_layer_call_fn_18069mVXW?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€2

 
p

 
™ "%К"
unknown€€€€€€€€€2≈
E__inference_sequential_layer_call_and_return_conditional_losses_17026|
VXW67EFTU@Ґ=
6Ґ3
)К&
embedding_input€€€€€€€€€2
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ≈
E__inference_sequential_layer_call_and_return_conditional_losses_17087|
VXW67EFTU@Ґ=
6Ґ3
)К&
embedding_input€€€€€€€€€2
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Љ
E__inference_sequential_layer_call_and_return_conditional_losses_17525s
VXW67EFTU7Ґ4
-Ґ*
 К
inputs€€€€€€€€€2
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Љ
E__inference_sequential_layer_call_and_return_conditional_losses_17977s
VXW67EFTU7Ґ4
-Ґ*
 К
inputs€€€€€€€€€2
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Я
*__inference_sequential_layer_call_fn_16315q
VXW67EFTU@Ґ=
6Ґ3
)К&
embedding_input€€€€€€€€€2
p 

 
™ "!К
unknown€€€€€€€€€Я
*__inference_sequential_layer_call_fn_16965q
VXW67EFTU@Ґ=
6Ґ3
)К&
embedding_input€€€€€€€€€2
p

 
™ "!К
unknown€€€€€€€€€Ц
*__inference_sequential_layer_call_fn_17197h
VXW67EFTU7Ґ4
-Ґ*
 К
inputs€€€€€€€€€2
p 

 
™ "!К
unknown€€€€€€€€€Ц
*__inference_sequential_layer_call_fn_17222h
VXW67EFTU7Ґ4
-Ґ*
 К
inputs€€€€€€€€€2
p

 
™ "!К
unknown€€€€€€€€€і
#__inference_signature_wrapper_17144М
VXW67EFTUKҐH
Ґ 
A™>
<
embedding_input)К&
embedding_input€€€€€€€€€2"1™.
,
dense_2!К
dense_2€€€€€€€€€
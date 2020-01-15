## What's Cooking?

Welcome to What's Cooking. I'm an amateur data scientist, and I set out to solve a text analysis classification problem from [Kaggle](https://www.kaggle.com/c/whats-cooking-kernels-only), using Julia language. The aim is to try and predict the cuisine of a recipe, given the ingredients in it.

## The Data
The data obtained from [Kaggle](https://www.kaggle.com/c/whats-cooking-kernels-only/data) was an array of dictionaries and looked like this:
```
39774-element Array{Any,1}:
 Dict{String,Any}("id"=>10259,"ingredients"=>Any["romaine lettuce", "black olives", "grape tomatoes", "garlic", "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles"],"cuisine"=>"greek")
 Dict{String,Any}("id"=>25693,"ingredients"=>Any["plain flour", "ground pepper", "salt", "tomatoes", "ground black pepper", "thyme", "eggs", "green tomatoes", "yellow corn meal", "milk", "vegetable oil"],"cuisine"=>"southern_us")
 Dict{String,Any}("id"=>20130,"ingredients"=>Any["eggs", "pepper", "salt", "mayonaise", "cooking oil", "green chilies", "grilled chicken breasts", "garlic powder", "yellow onion", "soy sauce", "butter", "chicken livers"],"cuisine"=>"filipino")
  â‹®
 Dict{String,Any}("id"=>2362,"ingredients"=>Any["green chile", "jalapeno chilies", "onions", "ground black pepper", "salt", "chopped cilantro fresh", "green bell pepper", "garlic", "white sugar", "roma tomatoes", "celery", "dried oregano"],"cuisine"=>"mexican")
```


Before loading the data we need to add our packages:
```julia
using CSV
using JSON
using TSne
using Plots
using LIBSVM
using Word2Vec
using Languages
using Distances
using DataFrames
using FreqTables
using TextAnalysis
using DecisionTree
```

Next let's load our data:
```julia
cd("/mnt/juliabox/Whats cooking")
s=open("train.json", "r")
data= JSON.parse(s::IO; dicttype=Dict, inttype=Int64);
```

## The Preprocessing

Going through the data, we notice a few things about the ingredients:-
 * We can simplify our data by converting all to lowercase, and removing stopwords such as 'of' and 'and'.
 * There are many verbs such as 'chopped' and 'crushed' that can be removed.
 * There are some recipes with just one ingredient, which is not useful.
 * There are many common ingredients such as salt, sugar, water, and oil which don't tell us anything about the cuisine. We can find these words by creating a frequency table to find the most common words.

The first and last step can be done using the TextAnalysis package.
```julia
for ii=1:length(data)
    ing=data[ii]["ingredients"]

    # Simplify by converting to lowercase, removing stopwords
    ing=convert(Array{String,1}, ing)
    sd = TokenDocument(ing)
    prepare!(sd, strip_case)
    prepare!(sd, strip_stopwords)
    simplified_ing=text(sd)

    # Remove verbs
    verbless_ing=[]
    b=split(simplified_ing)
    b2=""
    for j in b
        if endswith(j,"ed")== true && j!="red"
            filter!(e->e!=j,b)
        end
    end
    for j in b
        b2=b2*j*" "
    end
    verbless_ing=vcat(verbless_ing,b2)
    s_verbless_ing=[]
    for i in verbless_ing
        x= (rstrip(i))
        s_verbless_ing=vcat(s_verbless_ing,x)
    end

    # Remove recipes with jus one ingredient
    if length(data[i]["ingredients"])==1
        filter!(e->e!=data[i],data)
    end
end
```

## The Word2Vec Model

Now that our preprocessing is almost over, we can start thinking about our model. Word2Vec is a useful package that allows us to form vectors for all the words, based on the words that around it. This means, if implemented correctly, we should be able to create vectors that put cuisines close to their representative ingredients, such as 'Mexican' with 'salsa' and also put ingredients close to other similar ingredients, such as 'pasta' with 'parmesan'.
That sounds like a promising start. In order to form a Word2Vec model, we first need to form a corpus that has one recipe per line. We will be splitting up the ingredients into individual words. For each word, Word2Vec will be using a window of words around it, so it seems like a good idea to put the cuisine name in the middle of the recipe.
```julia
io=open("corpus.txt", "w");
for i=1:length(data)
    for j =1:length(data[i]["ingredients"])
        if j==div(length(data[i]["ingredients"]),2)
            print(io, data[i]["cuisine"],"\t")
        end
        for k in (split(data[i]["ingredients"][j]))
            print(io, k,"\t")
        end
    end
    print(io, "\n")
end
close(io)
```

Remember that we still have perform the last pre-processing step, that is to remove the frequent words. Let us read the corpus text to find the most frequent words.
```julia
txt = read("corpus.txt", String)
words = split(replace(txt, r"\P{L}"i => " "))
table = sort(freqtable(words); rev=true)
println(table[1:20])
```

Pick out some of these words and remove them:
```julia
 io=open("corpus.txt", "w");
 sd = Document("corpus.txt")
 close(io)
 io=open("new corpus.txt","w")
 remove_words!(sd,["salt","water","pepper","sugar","garlic","ground","fresh","butter","oil","sauce","onion","chicken","black","flour","powder","juice"])
 print(io,text(sd))
 close(io)
```

It's time to finally build our model using a window size of 9. Let us save the vectors in a file called "corpus vec.txt".
```julia
word2vec("new corpus.txt", "corpus vec.txt",window=9, min_count=2, verbose = true)
foodmodel=wordvectors("corpus vec.txt")
```

## The Visualisation

It's always a good idea to visualise some data, so let us visualise our cuisine vectors and see the patterns. For this we will reduce the vector dimensions from 100 to 2 using TSNE.

First, we need to create a matrix of cuisine vectors.
```julia
cuisine_matrix=zeros(100)
for i in cuisines
    c_vec=get_vector(foodmodel, i)
    cuisine_matrix=hcat(cuisine_matrix,c_vec)
end
cuisine_matrix=cuisine_matrix[:,2:21]
cuisine_matrix=cuisine_matrix'
```

Now to perform tsne and plot our results:
```julia
Y = tsne(cuisine_matrix[:,2:101], 2, 100, 5000, 2.25);
theplot = scatter(Y[:,1],Y[:,2],series_annotations=cuisines)
```
Here's the resulting plot:

![](https://github.com/AmitaVarma/What-s-cooking-/blob/master/plot.JPG)

From this we can see cuisines like russian, french, british and irish are grouped together, and so are cuisines like chinese, japanese, korean, thai, vietnamese, and filipino. This looks about right so let's proceed.

## Predictions on Word2Vec

How do we predict using these vectors? One easy way would be to find the average of all the ingredient vectors in a recipe and see which cuisine vector is the closest. Let's do this and find our training set accuracy. Remember, we have to do the same preprocessing steps on our training recipes.

First, let's form a vocabulary list, and a list of words we had removed before creating the model, so we can remove these same words from the recipes.
```julia
# To find the total list of words in the training set:
total_ing_list=[]
for i=1:length(data), j in split.(data[i]["ingredients"])
        total_ing_list=vcat(j,total_ing_list)
end
total_ing_list=unique(total_ing_list);
# To find the vocabulary in our model:
words = vocabulary(foodmodel)
# To find the words that are out of vocabulary:
ing_not_vocab=setdiff(total_ing_list,words);
```

```julia
correct=0
for ii=1:length(data)
    rec=data[ii]["ingredients"]
    new_rec=[]
    for j in split.(rec)
        new_rec=vcat(j,new_rec)
    end
    new_rec=setdiff(new_rec,ing_not_vocab)
    rec_vec=zeros(100)
    for i in new_rec
        a=split(i)
        for j in a
            ing_vec=get_vector(foodmodel,j)
            rec_vec+=ing_vec
        end
    end
    rec_vec=rec_vec/length(rec);
    pred=[]
    for i in cuisines
        cuisvec=get_vector(foodmodel, i)

        sim=cosine_dist(cuisvec, rec_vec)
        sim=1-sim
        predrow=hcat(sim,i)
        pred=vcat(pred,predrow)
    end
    pred=sortrows(pred,rev=true)
    fin_pred=pred[1,2]
    if fin_pred==data[ii]["cuisine"]
        correct+=1
    end
end
accuracy=(correct/length(data))*100
```

This gives us an accuracy of about 30%, which is okay considering we haven't actually used any classification techniques.

## The Decision Tree Model

Next, let's try using these vectors as features for a classification method. Lets start simple, with a decision tree. For the features, let's use the mean of the vectors in the recipe, as well as their max. Each of these vectors has a length of 100, giving us a total of 200 features.

Let us create these feature matrices after preprocessing.
```julia
f1_mat=zeros(100)
f2_mat=zeros(100)
for ii=1:length(data)
    rec=data[ii]["ingredients"]
    new_rec=[]
    for j in split.(rec)
        new_rec=vcat(j,new_rec)
    end
    new_rec=intersect(new_rec,words)

    rec_vec=zeros(100)
    for i in new_rec
        a=split(i)
        for j in a
            ing_vec=get_vector(foodmodel,j)
            rec_vec+=ing_vec
        end
    end

    rec_vec=rec_vec/length(rec);
    f1_mean=rec_vec;
    f1_mat=hcat(f1_mat,f1_mean)

    rec_mat=zeros(100,length(new_rec))
    for i=1:length(extra_rec)
        rec_mat[:,i]=get_vector(foodmodel,new_rec[i])
    end

    f2_max=maximum(rec_mat, dims=2)
    f2_mat=hcat(f2_mat,f2_max)

end
f1_mat=f1_mat[:,2:end];
f2_mat=f2_mat[:,2:end];
```

Here, `f1_mat` is the matrix of mean vectors of all the recipes, and `f2_mat` is the matrix of max vectors.

Let us join these to create our 200 features, and split our samples into a training and test set (x and xtest).
```julia
xtot=vcat(f1_mat,f2_mat);
xtot=xtot';
xtot=collect(xtot);
ntest=0.33*length(data);
xtest=[1:ntest,:]
x=[ntest+1:end,:]
```
Now to create our labels y, and similarly split them into ytest and y
```julia
ytot=Int64.([0])
for i=1:length(data)
    ind=findfirst(isequal(data[i]["cuisine"]),cuisines)
    ytot=vcat(ytot,ind)
end
ytest=ytot[2:ntest];
ytest=Float64.(ytest);
y=ytot[ntest+1:end];
y=Float64.(y)
```

To build our decision tree, we can try out some parameters to find which one gives us the best performance. I settled on the default parameters for building tree, and 0.8 for pruning threshold
```julia
modelt = build_tree(y,x);
modelt = prune_tree(modelt, 0.8)
```

This gives us:
```
Decision Tree
Leaves: 1336
Depth:  47
```

Let us predict the labels for xtest and check the confusion matrix to find the accuracy:
```julia
predst = apply_tree(modelt, xtest);
cm = confusion_matrix(ytest,predst)
```

Giving us:
```
Classes:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
Matrix:
Accuracy: 0.453515494835055
Kappa:    0.3918525511439742
```

With various parameters, this accuracy ranges from 45-50%, which is a pretty good improvement. But we can do better with a more complex model, such as an SVM.

## The C-SVC Model

Since this is a multi-class classification problem, let us use the C-SVC model from the LIBSVM library. This model has two parameters: gamma and cost, which can be tuned.

To find out the best values for the parameters, after a lot of trial and error, with this loop I managed to get the best performance.

```julia
gammas=[1.0,1.15,1.3]
costs=[2.7,2.8,2.9]
accuracies=zeros(3,3)
for i=1:3,j=1:3
    Î³=gammas[i];c=costs[j];
    svm_model = svmtrain(collect(x'),y,gamma=Î³,cost=c);
    (predicted_labels, decision_values) = svmpredict(svm_model, collect(xtest'));
    correct=sum(predicted_labels.==ytest)
    accuracies[i,j]=correct/length(ytest)
    a=accuracies[i,j];
    println("For gamma= $Î³ and cost= $c accuracy is $a")
end
```

This gives us a best performing accuracy for gamma= 1.0 and cost= 2.8, where the accuracy goes upto 74.5%. Considering that the best score on this dataset was 82.8%, I am pretty happy with my performance as a beginner.

The final step is to read the test data, and print the id number along with the prediction
```julia
s=open("test.json", "r")
results=zeros(length(test_data),2)
test_data= JSON.parse(s::IO; dicttype=Dict, inttype=Int64);
for i in 1:length(test_data)
    results[i,1]=test_data[i]["id"]
    rec=test_data[i]["ingredients"]
    new_rec=[]
    for j in split.(rec)
        new_rec=vcat(j,new_rec)
    end
    new_rec=intersect(new_rec,words)
    rec_vec=zeros(100)
    for j in new_rec
        ing_vec=get_vector(foodmodel,j)
        rec_vec+=ing_vec
    end
    rec_vec=rec_vec/length(rec);
    f1_mean=rec_vec;
    rec_mat=zeros(100,length(new_rec))
    for j=1:length(new_rec)
        rec_mat[:,i]=get_vector(foodmodel,j)
    end
    f2_max=maximum(rec_mat, dims=2)
    xi=vcat(f1_mean,f2_max)
    (predicted_labels, decision_values) = svmpredict(svm_model, collect(xi'))
    results[i,2]=predicted_labels
end
```

Thanks for reading through, and again, do give me feedback in the comments!

from __future__ import print_function
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA
import sqlite3
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class ranking:
    # Load the CSV file into a Pandas dataframe
    def __init__(self, data):
        n_feats = 2500
        # print(data)
        self.df = data.iloc[:n_feats]
        # Rename the "sypnopsis" column to "synopsis"
        # self.df = self.df.rename(columns={'sypnopsis': 'synopsis'})

        # Drop Duplicate names
        self.df.drop_duplicates(subset='Name', inplace=True)

        self.anime_id_to_index = {anime_id:index for index, anime_id in enumerate(self.df['MAL_ID'])}
        self.anime_name_to_id = {name:mid for name, mid in zip(self.df['Name'], self.df['MAL_ID'])}
        self.anime_id_to_name = {v:k for k,v in self.anime_name_to_id.items()}
        self.anime_name_to_index = {name:self.anime_id_to_index[self.anime_name_to_id[name]] for name in self.df['Name']}
        self.anime_index_to_name = {v:k for k,v in self.anime_name_to_index.items()}
        tfidf_vec = self.build_vectorizer(max_features=n_feats, stop_words="english")
        doc_by_vocab = np.empty([len(self.df), 5000])
        doc_by_vocab = tfidf_vec.fit_transform(self.df['synopsis'].values.astype('U'))
        self.doc_by_vocab = doc_by_vocab.toarray()
        self.word_to_index = tfidf_vec.vocabulary_

        # self.movie_sims_cos = np.array(matrix)
        # self.movie_sims_cos = self.build_movie_sims_cos(2500, self.anime_index_to_name, doc_by_vocab, self.anime_name_to_index, self.get_sim)

        docs_compressed, s, words_compressed = svds(doc_by_vocab, k=40)
        words_compressed = words_compressed.transpose()
        self.words_compressed_normed = normalize(words_compressed, axis = 1)
        self.docs_compressed_normed = normalize(docs_compressed)



    def build_vectorizer(self, max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
        """Returns a TfidfVectorizer object with the above preprocessing properties.
        
        Note: This function may log a deprecation warning. This is normal, and you
        can simply ignore it.
        
        Parameters
        ----------
        max_features : int
            Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
            constructer.
        stop_words : str
            Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
        max_df : float
            Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
        min_df : float
            Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
        norm : str
            Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

        Returns
        -------
        TfidfVectorizer
            A TfidfVectorizer object with the given parameters as its preprocessing properties.
        """
        # YOUR CODE HERE
        vectorizer = TfidfVectorizer(max_features = max_features, stop_words=stop_words, max_df=max_df, min_df=min_df, norm=norm)
        return vectorizer



    def get_sim(self, mov1, mov2, input_doc_mat, input_movie_name_to_index):
        """Returns a float giving the cosine similarity of 
        the two movie transcripts.
        
        Params: {mov1 (str): Name of the first movie.
                mov2 (str): Name of the second movie.
                input_doc_mat (numpy.ndarray): Term-document matrix of movie transcripts, where 
                        each row represents a document (movie transcript) and each column represents a term.
                movie_name_to_index (dict): Dictionary that maps movie names to the corresponding row index 
                        in the term-document matrix.}
        Returns: Float (Cosine similarity of the two movie transcripts.)
        """
        # YOUR CODE HERE
        index1 = input_movie_name_to_index[mov1]
        index2 = input_movie_name_to_index[mov2]
        arr1 = input_doc_mat[index1]
        arr2 = input_doc_mat[index2]
        numerator = np.dot(arr1,arr2)
        
        return numerator

    def build_movie_sims_cos(self, n_mov, movie_index_to_name, input_doc_mat, movie_name_to_index, input_get_sim_method):
        """Returns a movie_sims matrix of size (num_movies,num_movies) where for (i,j):
            [i,j] should be the cosine similarity between the movie with index i and the movie with index j
            
        Note: You should set values on the diagonal to 1
        to indicate that all movies are trivially perfectly similar to themselves.
        
        Params: {n_mov: Integer, the number of movies
                movie_index_to_name: Dictionary, a dictionary that maps movie index to name
                input_doc_mat: Numpy Array, a numpy array that represents the document-term matrix
                movie_name_to_index: Dictionary, a dictionary that maps movie names to index
                input_get_sim_method: Function, a function to compute cosine similarity}
        Returns: Numpy Array 
        """
        # YOUR CODE HERE
        movie_sims_matrix = np.zeros((n_mov, n_mov))
        
        for i in range(0, n_mov):
            for j in range(0, n_mov):

                sim_score = input_get_sim_method(movie_index_to_name[i], movie_index_to_name[j], input_doc_mat, movie_name_to_index)
                movie_sims_matrix[i][j] = sim_score

        # np.savetxt("matrix.csv", movie_sims_matrix, delimiter=",")
        return movie_sims_matrix



    def get_ranked_movies(self, mov, input_doc_mat, input_movie_name_to_index, df):
        """
        Return sorted rankings (most to least similar) of movies as 
        a list of two-element tuples, where the first element is the 
        movie name and the second element is the similarity score
        
        Params: {mov: String,
                matrix: np.ndarray}
        Returns: List<Tuple>
        """
        if mov not in self.anime_name_to_index:
            return[(df['Name'][i], 1) for i in range(0,len(df['Name']))]

        # # Get movie index from movie name
        # mov_idx = anime_name_to_index[mov]
        
        # score_lst = matrix[mov_idx]
        # mov_score_lst = [(anime_index_to_name[i], s) for i,s in enumerate(score_lst)]

        # Get list of similarity scores for movie
        mov_score_lst = []
        for index, row in df.iterrows(): 
            mov2 = row['Name']
            sim = 0
            if mov != mov2:
                sim = self.get_sim(mov, mov2, input_doc_mat, input_movie_name_to_index)
            mov_score_lst.append((row['Name'], sim))
            
        return mov_score_lst


    def closest_projects(self, word_in, project_repr_in, words_representation_in, df):
        if word_in not in self.word_to_index: return[(df['Name'][i], 1) for i in range(0,len(df['Name']))]
        sims = project_repr_in.dot(words_representation_in[self.word_to_index[word_in],:])
        return [(df['Name'][i],sims[i]) for i in range(0,len(sims))]

    def multiply_jac_sim(self, genres, df):
        # Get movie index from movie name
        arr = []
        A = set(genres)
        if len(A) == 0:
            return[(df['Name'][i], 1) for i in range(0,len(df['Name']))]
        for index, row in df.iterrows():
            l = row['Genres'].split(',')
            l = [s.strip() for s in l]
            B = set(l)
            jac_sim = 0
            # if(len(A.union(B)) > 0):
            #     jac_sim = len(A.intersection(B))/len(A.union(B)) 
            arr.append((row['Name'], len(A.intersection(B))))
            
        return arr
    
        
        
    def multiply_ratings(self, df):
        arr = []
        for index, row in df.iterrows():
            score = row['Score']
            try:
                score = float(score)
            except:
                score = 5
            arr.append((row['Name'], score))
        return arr


    def multiply_keywords(self, input_string, docs_compressed_normed, words_compressed_normed, df):
        word_array = input_string.split(", ") # split the string into an array of words
        word_array = [word.lower() for word in word_array] # convert all the words to lowercase
        keywords_array = [(df['Name'][i], 1) for i in range(0,len(df['Name']))]
        for w in word_array:
            word = self.closest_projects(w, docs_compressed_normed, words_compressed_normed, df)
            keywords_array = [(a[0], a[1]*b[1]) for a, b in zip(keywords_array, word)]
        return keywords_array

    def set_bottom_third(self, arr):
        # get the 33% mark value
        vals = [x[1] for x in arr]
        bottom_third_mark = sorted(vals)[int(len(vals) * 0.33)]
        
        # set the entries with the bottom 20% of values to the 20% value
        for i in range(len(arr)):
            if arr[i][1] < bottom_third_mark:
                arr[i] = (arr[i][0], bottom_third_mark)
        
        return arr

    def get_ranking(self, anime, genres, keywords):
        title_ranking = self.set_bottom_third(self.get_ranked_movies(anime, self.doc_by_vocab, self.anime_name_to_index, self.df))
        genre_ranking = self.multiply_jac_sim(genres, self.df)
        score_ranking = self.multiply_ratings(self.df) 
        keyword_ranking = self.multiply_keywords(keywords, self.docs_compressed_normed, self.words_compressed_normed, self.df)
        

        # Multiply the tuples in each list together
        product = [(a[0], a[1]*b[1]*c[1]*d[1]) for a, b, c, d in zip(title_ranking, genre_ranking, score_ranking, keyword_ranking)]

        sum_of_second_elements = 0
        for tuple in product:
            sum_of_second_elements += tuple[1]


        result = sorted(product, key=lambda x: x[1], reverse=True)
        for i, tup in enumerate(result):
            result[i] = (tup[0], self.df.loc[self.df['Name'] == tup[0], 'synopsis'].iloc[0], self.df.loc[self.df['Name'] == tup[0], 'image_url'].iloc[0])

        return result[:10]  
    
def main():
    anime = 'Cowboy Bebop'
    genres = ['Action', 'Fantasy']
    keywords = ''
    data = pd.read_csv('../data/output.csv')
    # matrix = pd.read_csv('../data/matrix.csv')
    r = ranking(data)
    results = r.get_ranking(anime, genres, keywords)
    print(results)

# main()
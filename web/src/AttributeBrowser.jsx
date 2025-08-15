import React, { useState, useEffect } from 'react';

const AttributeBrowser = ({ onAttributeSelect }) => {
  const [collections, setCollections] = useState({});
  const [favorites, setFavorites] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [suggestions, setSuggestions] = useState({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('collections');

  const safeFetch = async (url, options = {}) => {
    const response = await fetch(`http://localhost:8192${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  };

  useEffect(() => {
    loadCollections();
    loadFavorites();
    loadSuggestions();
  }, []);

  const loadCollections = async () => {
    try {
      const response = await safeFetch('/attributes/collections');
      const data = await response.json();
      setCollections(data.collections || []);
    } catch (error) {
      console.error('Failed to load collections:', error);
    }
  };

  const loadFavorites = async () => {
    try {
      const response = await safeFetch('/attributes/favorites');
      const data = await response.json();
      setFavorites(data.favorites || []);
    } catch (error) {
      console.error('Failed to load favorites:', error);
    }
  };

  const loadSuggestions = async () => {
    try {
      const categories = ['namespace', 'persona', 'style'];
      const suggestionData = {};
      
      for (const category of categories) {
        const response = await safeFetch(`/attributes/suggestions/${category}`);
        const data = await response.json();
        suggestionData[category] = data.suggestions || {};
      }
      
      setSuggestions(suggestionData);
    } catch (error) {
      console.error('Failed to load suggestions:', error);
    }
  };

  const searchAttributes = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await safeFetch('/attributes/search', {
        method: 'POST',
        body: JSON.stringify({
          query: searchQuery,
          category: selectedCategory === 'all' ? null : selectedCategory,
          limit: 20
        })
      });
      const data = await response.json();
      setSearchResults(data.attributes || []);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleFavorite = async (attributeName, isFavorite) => {
    try {
      const endpoint = isFavorite ? '/attributes/favorites/remove' : '/attributes/favorites/add';
      await safeFetch(endpoint, {
        method: 'POST',
        body: JSON.stringify({ attribute_name: attributeName })
      });
      
      // Refresh favorites
      await loadFavorites();
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  const AttributeCard = ({ attribute, showCategory = false }) => (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: 8,
      padding: 12,
      margin: '8px 0',
      background: attribute.is_favorite ? '#fff3e0' : '#fff'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 5px 0', color: '#333' }}>
            {attribute.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </h4>
          {showCategory && (
            <span style={{ 
              fontSize: 12, 
              color: '#666', 
              background: '#f0f0f0', 
              padding: '2px 6px', 
              borderRadius: 4,
              marginRight: 8
            }}>
              {attribute.category} • {attribute.subcategory}
            </span>
          )}
          <p style={{ margin: '5px 0', fontSize: 14, color: '#666' }}>
            {attribute.description}
          </p>
          {attribute.tags && (
            <div style={{ marginTop: 8 }}>
              {attribute.tags.slice(0, 3).map(tag => (
                <span key={tag} style={{
                  fontSize: 11,
                  color: '#555',
                  background: '#e8f5e8',
                  padding: '2px 6px',
                  borderRadius: 3,
                  marginRight: 4
                }}>
                  {tag}
                </span>
              ))}
            </div>
          )}
          {attribute.positive_examples && (
            <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
              <em>"{attribute.positive_examples[0]}"</em>
            </div>
          )}
        </div>
        <div style={{ marginLeft: 12 }}>
          <button
            onClick={() => toggleFavorite(attribute.name, attribute.is_favorite)}
            style={{
              background: 'none',
              border: 'none',
              fontSize: 18,
              cursor: 'pointer',
              color: attribute.is_favorite ? '#ff6b35' : '#ccc'
            }}
          >
            ★
          </button>
          <button
            onClick={() => onAttributeSelect && onAttributeSelect(attribute)}
            style={{
              marginTop: 8,
              padding: '4px 8px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              fontSize: 12,
              cursor: 'pointer'
            }}
          >
            Use
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h2>🎨 Attribute Library</h2>
      
      {/* Tab Navigation */}
      <div style={{ borderBottom: '2px solid #ddd', marginBottom: 20 }}>
        {['collections', 'favorites', 'search', 'suggestions'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '10px 20px',
              background: activeTab === tab ? '#007bff' : '#f8f9fa',
              color: activeTab === tab ? 'white' : '#333',
              border: 'none',
              borderRadius: '8px 8px 0 0',
              marginRight: 5,
              cursor: 'pointer'
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Collections Tab */}
      {activeTab === 'collections' && (
        <div>
          <h3>📚 Browse Collections</h3>
          {collections.map((collection) => (
            <div key={collection.id} style={{ marginBottom: 30 }}>
              <h4 style={{ color: '#007bff', marginBottom: 10 }}>
                {collection.name} ({collection.attributes.length} attributes)
              </h4>
              <p style={{ color: '#666', marginBottom: 15 }}>{collection.description}</p>
              <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                {collection.attributes.map(attr => (
                  <AttributeCard key={attr.name} attribute={attr} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Favorites Tab */}
      {activeTab === 'favorites' && (
        <div>
          <h3>⭐ Your Favorites ({favorites.length})</h3>
          {favorites.length === 0 ? (
            <p style={{ color: '#666', fontStyle: 'italic' }}>
              No favorites yet. Star some attributes to add them here!
            </p>
          ) : (
            favorites.map(attr => (
              <AttributeCard key={attr.name} attribute={{...attr, is_favorite: true}} showCategory={true} />
            ))
          )}
        </div>
      )}

      {/* Search Tab */}
      {activeTab === 'search' && (
        <div>
          <h3>🔍 Search Attributes</h3>
          <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search attributes..."
              style={{
                flex: 1,
                padding: 10,
                border: '1px solid #ddd',
                borderRadius: 4
              }}
              onKeyPress={(e) => e.key === 'Enter' && searchAttributes()}
            />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              style={{ padding: 10, border: '1px solid #ddd', borderRadius: 4 }}
            >
              <option value="all">All Categories</option>
              <option value="namespace">Namespace</option>
              <option value="persona">Persona</option>
              <option value="style">Style</option>
            </select>
            <button
              onClick={searchAttributes}
              disabled={loading || !searchQuery.trim()}
              style={{
                padding: '10px 20px',
                background: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer'
              }}
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
          
          {searchResults.length > 0 && (
            <div>
              <h4>Results ({searchResults.length})</h4>
              {searchResults.map(attr => (
                <AttributeCard key={attr.name} attribute={attr} showCategory={true} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Suggestions Tab */}
      {activeTab === 'suggestions' && (
        <div>
          <h3>💡 Curated Suggestions</h3>
          {Object.entries(suggestions).map(([category, categoryData]) => (
            <div key={category} style={{ marginBottom: 30 }}>
              <h4 style={{ color: '#007bff', textTransform: 'capitalize' }}>
                {category} Attributes
              </h4>
              {Object.entries(categoryData).map(([level, attrs]) => (
                <div key={level} style={{ marginBottom: 20 }}>
                  <h5 style={{ color: '#666', textTransform: 'capitalize', marginBottom: 10 }}>
                    {level.replace('_', ' ')} ({attrs.length})
                  </h5>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                    {attrs.map(attr => (
                      <div key={attr.name} style={{
                        border: '1px solid #ddd',
                        borderRadius: 8,
                        padding: 10,
                        flex: '0 1 calc(50% - 5px)',
                        minWidth: 250,
                        background: attr.is_favorite ? '#fff3e0' : '#fff'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <strong>{attr.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong>
                            <p style={{ margin: '5px 0', fontSize: 13, color: '#666' }}>
                              {attr.description}
                            </p>
                          </div>
                          <div>
                            <button
                              onClick={() => toggleFavorite(attr.name, attr.is_favorite)}
                              style={{
                                background: 'none',
                                border: 'none',
                                fontSize: 16,
                                cursor: 'pointer',
                                color: attr.is_favorite ? '#ff6b35' : '#ccc'
                              }}
                            >
                              ★
                            </button>
                            <button
                              onClick={() => onAttributeSelect && onAttributeSelect(attr)}
                              style={{
                                marginLeft: 5,
                                padding: '2px 6px',
                                background: '#007bff',
                                color: 'white',
                                border: 'none',
                                borderRadius: 3,
                                fontSize: 11,
                                cursor: 'pointer'
                              }}
                            >
                              Use
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AttributeBrowser;
import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardMedia,
  CardActions,
  Grid,
  TextField,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  InputAdornment,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Search as SearchIcon,
  Sell as SellIcon,
  ShoppingCart as PurchaseIcon,
  AddPhotoAlternate as ImageIcon,
  Place as PlaceIcon,
  AttachMoney as MoneyIcon,
  List as ListIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:5001/api';

// Contact: only digits count; must be 10 digits
function isContact10Digits(value) {
  const digits = (value || '').replace(/\D/g, '');
  return digits.length === 10;
}

// Basic valid email format
const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
function isValidEmail(value) {
  return value && typeof value === 'string' && EMAIL_REGEX.test(value.trim());
}

function imageUrl(filename) {
  if (!filename) return null;
  return `${API_BASE_URL.replace('/api', '')}/api/images/${filename}`;
}

function MarketplacePage() {
  const [listings, setListings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Search filters
  const [searchName, setSearchName] = useState('');
  const [searchCostMin, setSearchCostMin] = useState('');
  const [searchCostMax, setSearchCostMax] = useState('');
  const [searchArea, setSearchArea] = useState('');

  // Sell form
  const [sellOpen, setSellOpen] = useState(false);
  const [sellRiceName, setSellRiceName] = useState('');
  const [sellCost, setSellCost] = useState('');
  const [sellArea, setSellArea] = useState('');
  const [sellImage, setSellImage] = useState(null);
  const [sellContact, setSellContact] = useState('');
  const [sellEmail, setSellEmail] = useState('');
  const [selling, setSelling] = useState(false);

  // Purchase dialog
  const [purchaseListing, setPurchaseListing] = useState(null);
  const [purchaseContact, setPurchaseContact] = useState('');
  const [purchaseEmail, setPurchaseEmail] = useState('');
  const [purchasing, setPurchasing] = useState(false);

  // Purchase list (who purchased which rice and when)
  const [purchases, setPurchases] = useState([]);
  const [purchasesLoading, setPurchasesLoading] = useState(false);

  const buildSearchParams = () => {
    const params = new URLSearchParams();
    if (searchName.trim()) params.set('name', searchName.trim());
    if (searchCostMin !== '') params.set('cost_min', searchCostMin);
    if (searchCostMax !== '') params.set('cost_max', searchCostMax);
    if (searchArea.trim()) params.set('area', searchArea.trim());
    return params.toString();
  };

  const loadListings = async () => {
    setLoading(true);
    setError(null);
    try {
      const query = buildSearchParams();
      const url = query ? `${API_BASE_URL}/listings?${query}` : `${API_BASE_URL}/listings`;
      const response = await fetch(url);
      const data = await response.json();
      if (data.success) {
        setListings(data.listings || []);
      } else {
        setError('Failed to load listings');
      }
    } catch (err) {
      setError(`Error: ${err.message}. Ensure API is running on port 5001.`);
      setListings([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadListings();
    loadPurchaseList();
  }, []);

  const handleSearch = () => {
    loadListings();
  };

  const loadPurchaseList = async () => {
    setPurchasesLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/purchase-list`);
      const data = await response.json();
      if (data.success) setPurchases(data.purchases || []);
    } catch {
      setPurchases([]);
    } finally {
      setPurchasesLoading(false);
    }
  };

  const handleSellSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (!sellRiceName.trim()) {
      setError('Rice name is required');
      return;
    }
    const costNum = parseFloat(sellCost);
    if (sellCost === '' || sellCost === null || isNaN(costNum) || costNum < 0) {
      setError('Cost is required and must be 0 or more');
      return;
    }
    if (!sellArea.trim()) {
      setError('Area (where rice is grown) is required');
      return;
    }
    if (!sellImage) {
      setError('Image upload is required');
      return;
    }
    if (!sellContact.trim()) {
      setError('Contact number is required');
      return;
    }
    if (!isContact10Digits(sellContact)) {
      setError('Contact number must be exactly 10 digits');
      return;
    }
    if (!sellEmail.trim()) {
      setError('Email is required');
      return;
    }
    if (!isValidEmail(sellEmail)) {
      setError('Please enter a valid email address');
      return;
    }
    setSelling(true);
    try {
      const formData = new FormData();
      formData.append('rice_name', sellRiceName.trim());
      formData.append('cost', String(costNum));
      formData.append('area', sellArea.trim());
      formData.append('seller_contact', sellContact.trim());
      formData.append('seller_email', sellEmail.trim());
      if (sellImage) formData.append('image', sellImage);
      const response = await fetch(`${API_BASE_URL}/listings`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        setSuccessMsg('Listing created. Your rice is now listed for sale.');
        setSellOpen(false);
        setSellRiceName('');
        setSellCost('');
        setSellArea('');
        setSellImage(null);
        setSellContact('');
        setSellEmail('');
        loadListings();
        loadPurchaseList();
        setTimeout(() => setSuccessMsg(null), 4000);
      } else {
        setError(data.error || 'Failed to create listing');
      }
    } catch (err) {
      setError(err.message || 'Failed to create listing');
    } finally {
      setSelling(false);
    }
  };

  const handlePurchaseSubmit = async (e) => {
    e.preventDefault();
    if (!purchaseListing) return;
    if (!purchaseContact.trim()) {
      setError('Contact number is required');
      return;
    }
    if (!purchaseEmail.trim()) {
      setError('Email is required');
      return;
    }
    setPurchasing(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/listings/${purchaseListing.id}/purchase`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contact: purchaseContact.trim(), email: purchaseEmail.trim() }),
      });
      const data = await response.json();
      if (data.success) {
        setSuccessMsg('Purchase interest recorded. Seller will contact you.');
        setPurchaseListing(null);
        setPurchaseContact('');
        setPurchaseEmail('');
        loadPurchaseList();
        setTimeout(() => setSuccessMsg(null), 4000);
      } else {
        setError(data.error || 'Failed to submit');
      }
    } catch (err) {
      setError(err.message || 'Failed to submit');
    } finally {
      setPurchasing(false);
    }
  };

  return (
    <Container maxWidth="xl">
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
        Rice Marketplace
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Search rice by name, cost, and area. Sellers and buyers must provide contact number and email.
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {successMsg && (
        <Alert severity="success" onClose={() => setSuccessMsg(null)} sx={{ mb: 2 }}>
          {successMsg}
        </Alert>
      )}

      {/* Search bar */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <SearchIcon /> Search rice
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              size="small"
              label="Rice name"
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
              placeholder="e.g. Basmati"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              fullWidth
              size="small"
              type="number"
              label="Min cost (₹)"
              value={searchCostMin}
              onChange={(e) => setSearchCostMin(e.target.value)}
              InputProps={{ startAdornment: <InputAdornment position="start"><MoneyIcon fontSize="small" /></InputAdornment> }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              fullWidth
              size="small"
              type="number"
              label="Max cost (₹)"
              value={searchCostMax}
              onChange={(e) => setSearchCostMax(e.target.value)}
              InputProps={{ startAdornment: <InputAdornment position="start"><MoneyIcon fontSize="small" /></InputAdornment> }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              size="small"
              label="Area (growth region)"
              value={searchArea}
              onChange={(e) => setSearchArea(e.target.value)}
              placeholder="e.g. Punjab, Kerala"
              InputProps={{ startAdornment: <InputAdornment position="start"><PlaceIcon fontSize="small" /></InputAdornment> }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <Button variant="contained" onClick={handleSearch} startIcon={<SearchIcon />} fullWidth>
              Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Sell rice button */}
      <Box sx={{ mb: 3 }}>
        <Button
          variant="outlined"
          color="primary"
          startIcon={<SellIcon />}
          onClick={() => setSellOpen(true)}
        >
          Sell rice (add listing)
        </Button>
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Listings grid */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      ) : listings.length === 0 ? (
        <Typography color="text.secondary">No listings match your search. Try different filters or add a new listing.</Typography>
      ) : (
        <Grid container spacing={3}>
          {listings.map((listing, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={listing.id}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  {listing.image_filenames && listing.image_filenames[0] ? (
                    <CardMedia
                      component="img"
                      height="160"
                      image={imageUrl(listing.image_filenames[0])}
                      alt={listing.rice_name}
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  ) : (
                    <Box sx={{ height: 160, bgcolor: 'action.hover', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <ImageIcon sx={{ fontSize: 48, color: 'text.disabled' }} />
                    </Box>
                  )}
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {listing.rice_name || 'Unnamed'}
                    </Typography>
                    <Typography variant="body2" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <MoneyIcon fontSize="small" /> ₹{Number(listing.cost).toLocaleString()}
                    </Typography>
                    {listing.area && (
                      <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                        <PlaceIcon fontSize="small" /> {listing.area}
                      </Typography>
                    )}
                  </CardContent>
                  <CardActions>
                    <Button
                      size="small"
                      color="primary"
                      startIcon={<PurchaseIcon />}
                      onClick={() => {
                        setPurchaseListing(listing);
                        setPurchaseContact('');
                        setPurchaseEmail('');
                      }}
                    >
                      Purchase
                    </Button>
                  </CardActions>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Sell rice dialog */}
      <Dialog open={sellOpen} onClose={() => setSellOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Sell rice — add listing</DialogTitle>
        <form onSubmit={handleSellSubmit}>
          <DialogContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              All fields are required: rice name, cost, area, image, and your contact (10 digits) and email.
            </Typography>
            <TextField fullWidth label="Rice name *" value={sellRiceName} onChange={(e) => setSellRiceName(e.target.value)} margin="dense" required />
            <TextField fullWidth type="number" label="Cost (₹) *" value={sellCost} onChange={(e) => setSellCost(e.target.value)} margin="dense" inputProps={{ min: 0, step: 0.01 }} required />
            <TextField fullWidth label="Area (where rice is grown) *" value={sellArea} onChange={(e) => setSellArea(e.target.value)} margin="dense" placeholder="e.g. Punjab, Kerala" required />
            <Box sx={{ mt: 2, mb: 1 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>Image * (required)</Typography>
              <Button variant="outlined" component="label" startIcon={<ImageIcon />} size="small" color={sellImage ? 'primary' : 'inherit'}>
                {sellImage ? sellImage.name : 'Choose image'}
                <input type="file" accept="image/*" hidden onChange={(e) => setSellImage(e.target.files?.[0] || null)} />
              </Button>
              {!sellImage && <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>PNG, JPG, JPEG, GIF or BMP</Typography>}
            </Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" gutterBottom>Selling contact (required)</Typography>
            <TextField fullWidth label="Contact number * (10 digits)" value={sellContact} onChange={(e) => setSellContact(e.target.value)} margin="dense" placeholder="e.g. 9876543210" required error={!!sellContact && !isContact10Digits(sellContact)} helperText={sellContact && !isContact10Digits(sellContact) ? 'Must be exactly 10 digits' : ''} inputProps={{ maxLength: 14 }} />
            <TextField fullWidth type="email" label="Email *" value={sellEmail} onChange={(e) => setSellEmail(e.target.value)} margin="dense" placeholder="seller@example.com" required error={!!sellEmail && !isValidEmail(sellEmail)} helperText={sellEmail && !isValidEmail(sellEmail) ? 'Enter a valid email' : ''} />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSellOpen(false)}>Cancel</Button>
            <Button type="submit" variant="contained" disabled={selling || !sellRiceName.trim() || sellCost === '' || isNaN(parseFloat(sellCost)) || parseFloat(sellCost) < 0 || !sellArea.trim() || !sellImage || !isContact10Digits(sellContact) || !isValidEmail(sellEmail)}>
              {selling ? <CircularProgress size={24} /> : 'Create listing'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Purchase interest dialog */}
      <Dialog open={!!purchaseListing} onClose={() => setPurchaseListing(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Purchase — {purchaseListing?.rice_name}</DialogTitle>
        <form onSubmit={handlePurchaseSubmit}>
          <DialogContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Contact number and email are required so the seller can reach you.
            </Typography>
            <TextField fullWidth label="Contact number *" value={purchaseContact} onChange={(e) => setPurchaseContact(e.target.value)} margin="dense" placeholder="e.g. +91 98765 43210" required />
            <TextField fullWidth type="email" label="Email *" value={purchaseEmail} onChange={(e) => setPurchaseEmail(e.target.value)} margin="dense" placeholder="buyer@example.com" required />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setPurchaseListing(null)}>Cancel</Button>
            <Button type="submit" variant="contained" disabled={purchasing || !purchaseContact.trim() || !purchaseEmail.trim()}>
              {purchasing ? <CircularProgress size={24} /> : 'Submit'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Purchase list: who purchased which rice and when */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <ListIcon /> Purchase list
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          All purchase interests: who purchased which rice and when. Sellers and buyers can use this to follow up.
        </Typography>
        {purchasesLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={32} />
          </Box>
        ) : purchases.length === 0 ? (
          <Typography color="text.secondary">No purchases yet. Purchase interests will appear here after buyers submit.</Typography>
        ) : (
          <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Rice</strong></TableCell>
                  <TableCell><strong>Area</strong></TableCell>
                  <TableCell><strong>Cost (₹)</strong></TableCell>
                  <TableCell><strong>Buyer contact</strong></TableCell>
                  <TableCell><strong>Buyer email</strong></TableCell>
                  <TableCell><strong>Seller contact</strong></TableCell>
                  <TableCell><strong>Seller email</strong></TableCell>
                  <TableCell><strong>Purchased at</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {purchases.map((p, i) => (
                  <TableRow key={i}>
                    <TableCell>{p.rice_name}</TableCell>
                    <TableCell>{p.area || '—'}</TableCell>
                    <TableCell>{Number(p.cost).toLocaleString()}</TableCell>
                    <TableCell>{p.buyer_contact}</TableCell>
                    <TableCell>{p.buyer_email}</TableCell>
                    <TableCell>{p.seller_contact}</TableCell>
                    <TableCell>{p.seller_email}</TableCell>
                    <TableCell>{p.purchased_at ? new Date(p.purchased_at).toLocaleString() : '—'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>
    </Container>
  );
}

export default MarketplacePage;

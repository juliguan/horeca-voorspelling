// Vooruitzicht — Rooster (mobiele app, eerste opzet)
//
// Praat met de FastAPI-backend in ../api.py (POST /rooster): kassa-CSV
// uploaden, locatie + omzet-norm invullen, roosteradvies per dagdeel terug.
//
// API_BASE_URL wijst nu naar 127.0.0.1 -- dat werkt in de iOS Simulator op
// deze Mac (die deelt localhost met de host), maar NIET op een fysiek
// device of de Android-emulator (die heeft 10.0.2.2 nodig) en zeker niet
// op een telefoon buiten dit netwerk -- daarvoor moet api.py eerst ergens
// online staan, net als de Streamlit-app op Streamlit Cloud staat.
import { useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';

const API_BASE_URL = Platform.select({
  ios: 'http://127.0.0.1:8000',
  android: 'http://10.0.2.2:8000',
  default: 'http://127.0.0.1:8000',
});
const ACCENT = '#D97757';
const DAGDELEN = ['ochtend', 'lunch', 'middag', 'diner', 'avond'];

export default function App() {
  const [bestand, setBestand] = useState(null);
  const [locatie, setLocatie] = useState('Cafe Laurierboom, Amsterdam');
  const [norm, setNorm] = useState('100');
  const [laden, setLaden] = useState(false);
  const [fout, setFout] = useState(null);
  const [resultaat, setResultaat] = useState(null);

  const kiesBestand = async () => {
    setFout(null);
    const res = await DocumentPicker.getDocumentAsync({
      type: ['text/csv', 'text/comma-separated-values', 'public.comma-separated-values-text'],
      copyToCacheDirectory: true,
    });
    if (res.canceled) return;
    setBestand(res.assets[0]);
    setResultaat(null);
  };

  const berekenAdvies = async () => {
    if (!bestand) return;
    setLaden(true);
    setFout(null);
    setResultaat(null);
    try {
      const form = new FormData();
      form.append('orders', {
        uri: bestand.uri,
        name: bestand.name || 'orders.csv',
        type: 'text/csv',
      });
      form.append('locatie', locatie);
      form.append('norm_omzet_per_uur', norm || '100');

      const response = await fetch(`${API_BASE_URL}/rooster`, {
        method: 'POST',
        body: form,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      setResultaat(data);
    } catch (e) {
      setFout(e.message || String(e));
    } finally {
      setLaden(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.wordmark}>Vooruitzicht</Text>
        <Text style={styles.subtitle}>Rooster</Text>
      </View>

      <View style={styles.veld}>
        <Text style={styles.label}>locatie (voor het weer)</Text>
        <TextInput style={styles.input} value={locatie} onChangeText={setLocatie} />
      </View>
      <View style={styles.veld}>
        <Text style={styles.label}>omzet-norm per gewerkt uur (€)</Text>
        <TextInput style={styles.input} value={norm} onChangeText={setNorm} keyboardType="decimal-pad" />
      </View>

      <TouchableOpacity style={styles.knopSecundair} onPress={kiesBestand}>
        <Text style={styles.knopSecundairTekst}>
          {bestand ? `📄 ${bestand.name}` : '📄 kies kassabestand (CSV)'}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.knop, !bestand && styles.knopUit]}
        onPress={berekenAdvies}
        disabled={!bestand || laden}
      >
        {laden ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.knopTekst}>bereken roosteradvies</Text>
        )}
      </TouchableOpacity>

      {fout && <Text style={styles.foutTekst}>⚠️ {fout}</Text>}

      {resultaat && (
        <>
          <Text style={styles.samenvatting}>
            testperiode {resultaat.test_periode.start} t/m {resultaat.test_periode.eind} (
            {resultaat.test_periode.aantal_dagen} dagen) · gem.{' '}
            {resultaat.weer_samenvatting.gem_temp_c}°C · {resultaat.weer_samenvatting.regendagen} regendagen
          </Text>
          <View style={styles.tabelHeader}>
            <Text style={[styles.cel, styles.celDatum, styles.tabelHeaderTekst]}>datum</Text>
            {DAGDELEN.map((d) => (
              <Text key={d} style={[styles.cel, styles.tabelHeaderTekst]}>
                {d.slice(0, 3)}
              </Text>
            ))}
          </View>
          <FlatList
            data={resultaat.advies}
            keyExtractor={(item) => item.datum}
            renderItem={({ item }) => (
              <View style={styles.tabelRij}>
                <Text style={[styles.cel, styles.celDatum]}>{item.datum}</Text>
                {DAGDELEN.map((d) => (
                  <Text key={d} style={styles.cel}>
                    {item[d]}
                  </Text>
                ))}
              </View>
            )}
          />
        </>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#141313', paddingHorizontal: 20, paddingTop: 12 },
  header: { flexDirection: 'row', alignItems: 'baseline', gap: 10, marginBottom: 20 },
  wordmark: { fontSize: 26, fontWeight: '700', color: '#EDEAE5' },
  subtitle: { fontSize: 16, color: ACCENT, fontWeight: '600' },
  veld: { marginBottom: 12 },
  label: { color: '#8B8680', fontSize: 12, marginBottom: 4 },
  input: {
    backgroundColor: '#1F1E1D',
    color: '#EDEAE5',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 15,
    borderWidth: 1,
    borderColor: '#2E2C2A',
  },
  knopSecundair: {
    backgroundColor: '#1F1E1D',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 4,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#2E2C2A',
  },
  knopSecundairTekst: { color: '#EDEAE5', fontSize: 14 },
  knop: { backgroundColor: ACCENT, borderRadius: 10, paddingVertical: 14, alignItems: 'center' },
  knopUit: { opacity: 0.4 },
  knopTekst: { color: '#fff', fontWeight: '700', fontSize: 15 },
  foutTekst: { color: '#E5484D', marginTop: 14 },
  samenvatting: { color: '#8B8680', fontSize: 12, marginTop: 18, marginBottom: 8 },
  tabelHeader: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: '#2E2C2A', paddingBottom: 6 },
  tabelHeaderTekst: { color: ACCENT, fontWeight: '700', fontSize: 11, textTransform: 'uppercase' },
  tabelRij: {
    flexDirection: 'row',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#1F1E1D',
  },
  cel: { flex: 1, color: '#EDEAE5', fontSize: 13, textAlign: 'center' },
  celDatum: { flex: 1.6, textAlign: 'left', color: '#EDEAE5' },
});
